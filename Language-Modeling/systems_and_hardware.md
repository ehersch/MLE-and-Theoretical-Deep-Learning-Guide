# GPU Architecture and Systems

Understanding GPU hardware is essential for writing efficient deep learning code. The gap between theoretical FLOP/s and achieved FLOP/s is often 10–100×. Closing that gap requires understanding the memory hierarchy, parallelism model, and how to write code that maps well to GPU execution.

---

## GPU execution model

A GPU is a massively parallel processor with a different philosophy than a CPU:

| | CPU | GPU |
|---|---|---|
| Cores | 8–128 | 10,000–16,000 (CUDA cores) |
| Clock speed | ~4 GHz | ~1.5–2 GHz |
| Cache | Large (MB) | Small (KB per SM) |
| Thread model | Few heavyweight threads | Millions of lightweight threads |
| Latency vs throughput | Latency-optimized | Throughput-optimized |

GPUs hide memory latency by switching between **warps** (groups of 32 threads) when one warp stalls on a memory access.

### Streaming Multiprocessors (SMs)

An A100 has **108 SMs**. Each SM contains:
- 64 FP32 CUDA cores (4,096 tensor core FLOP/s per cycle at BF16)
- 4 tensor core units (for matrix multiply)
- 256 KB shared memory / L1 cache
- 32 warp schedulers

Threads are organized as: **grid → blocks → threads**, where each block runs on one SM.

### Tensor Cores

Specialized matrix multiply units that operate on tiles (e.g., 16×16×16 BF16 → FP32). A100 tensor cores deliver **312 TFLOP/s** (BF16) vs ~20 TFLOP/s for FP32 CUDA cores. Almost all LLM compute happens on tensor cores.

---

## Memory hierarchy

```
┌─────────────────────────────────────────────────────┐
│  HBM (High-Bandwidth Memory) — 80 GB, 2 TB/s        │  ← main GPU RAM
│  ↑                                                   │
│  L2 cache — 40 MB, ~5 TB/s                          │
│  ↑                                                   │
│  L1 / Shared memory — 256 KB per SM, ~20 TB/s       │  ← programmer-managed
│  ↑                                                   │
│  Registers — 256 KB per SM, instantaneous           │
└─────────────────────────────────────────────────────┘
```

**The key insight:** compute is fast, memory is slow. Moving data from HBM to SM (and back) is often the bottleneck.

| Level | Bandwidth | Latency |
|-------|-----------|---------|
| Registers | ~20 TB/s | 1 cycle |
| Shared memory | ~20 TB/s | ~30 cycles |
| L2 cache | ~5 TB/s | ~200 cycles |
| HBM (A100) | ~2 TB/s | ~500 cycles |
| PCIe (CPU↔GPU) | ~32 GB/s | μs |
| NVLink (GPU↔GPU) | ~600 GB/s | μs |

---

## Arithmetic Intensity and the Roofline Model

**Arithmetic intensity** of an operation = FLOPs / bytes transferred from memory:

$$I = \frac{\text{FLOPs}}{\text{Bytes}}$$

The **roofline model** says: performance is bounded by either compute or memory bandwidth:

$$\text{Achievable TFLOP/s} = \min\!\left(\text{Peak TFLOP/s},\ I \cdot \text{Bandwidth}\right)$$

**Ridge point** = the intensity where compute and memory bandwidth are equally limiting:

$$I^* = \frac{\text{Peak TFLOP/s}}{\text{Bandwidth}} = \frac{312 \text{ TFLOP/s}}{2 \text{ TB/s}} = 156 \text{ FLOP/byte}$$

Operations with $I < 156$ on A100 are **memory-bound**; those with $I > 156$ are **compute-bound**.

### Example: attention softmax

For a sequence of $T$ tokens, the softmax over attention logits:
- FLOPs: $O(T^2)$ (comparisons and exp)
- Bytes: read $T^2$ logits from HBM, write $T^2$ softmax values back
- Intensity ≈ $\sim 1$ FLOP/byte → extremely memory-bound

This is why FlashAttention (see [flash_attention.md](flash_attention.md)) is a massive win: it fuses operations to avoid repeated HBM round-trips.

### Example: matrix multiply

$C = AB$ where $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$:
- FLOPs: $2MNK$
- Bytes: $MK + KN + MN$
- For large square matrices: $I \approx N/2$

At $N=4096$: $I \approx 2048 \gg 156$ → compute-bound. Large matmuls saturate tensor cores.

---

## Profiling and memory accounting

### FLOPs per forward pass

For a transformer with $L$ layers, $d$ hidden, $d_{\text{ff}} = 4d$, sequence length $T$:

- Attention QKV projections: $6Td^2$ per layer
- Attention computation: $4T^2 d$ per layer
- FFN: $8Td^2$ per layer (two matmuls with $4d$ intermediate)
- Total per layer: $(14d + 4T)Td \approx 14Td^2$ (for $T \ll d$)

Total forward: $\approx 2 \times \text{params} \times T$ FLOPs (rule of thumb: ~6× for forward+backward)

**For LLaMA-7B training on 1T tokens:** $\approx 6 \times 7\times10^9 \times 10^{12} = 4.2 \times 10^{22}$ FLOPs

### Memory during training

For a model with $N$ parameters in BF16:

| Component | Bytes |
|-----------|-------|
| Weights (BF16) | $2N$ |
| Gradients (BF16) | $2N$ |
| Adam first moment (FP32) | $4N$ |
| Adam second moment (FP32) | $4N$ |
| Activations (recomputed) | varies |
| **Total** | **≥ 12N bytes** |

For 7B model: ≥ 84 GB → needs multiple GPUs.

---

## Writing GPU kernels: Triton

**Triton** is a Python-embedded DSL (by OpenAI) for writing custom GPU kernels. It operates at the **tile** level (blocks of elements), letting you express tiling strategies without CUDA's verbose thread indexing.

### Why write custom kernels?

PyTorch fuses some operations but not all. A naive softmax reads the input twice (once for max, once for exp+sum). A fused kernel does it in one pass. Same for RMSNorm, layer norm, rotary embedding, etc.

### Triton programming model

You write a **kernel function** that operates on one tile of data. The runtime launches many instances in parallel.

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                      # which tile am I?
    offsets = pid * BLOCK + tl.arange(0, BLOCK) # element indices for this tile
    mask = offsets < N                          # guard for last tile
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

# Launch: each program instance handles BLOCK elements
grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
add_kernel[grid](x, y, out, N, BLOCK=1024)
```

### Tiled matrix multiplication in Triton

The canonical example. Key idea: load tiles of A and B into shared memory, accumulate partial results, write output tile.

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Pointers to the tile in A and B
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)           # tensor core matmul
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Write output
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)
```

**Key optimization levers in Triton:**
- `BLOCK_M, BLOCK_N, BLOCK_K`: tile sizes (must be powers of 2; tune with `triton.autotune`)
- `num_stages`: pipeline depth (overlap HBM load with compute)
- `num_warps`: warps per block

### When to use Triton vs PyTorch

| Use PyTorch | Write Triton kernel |
|-------------|---------------------|
| Prototyping, not on critical path | Operation repeated millions of times |
| Standard ops (matmul, softmax) | Fused operations (flash attention, fused norm) |
| Simple elementwise ops | Memory-bound ops with complex fusion |

In practice: the LLM training stack uses Triton for attention (FlashAttention is Triton/CUDA), norms, and activations; PyTorch for everything else.

---

## Multi-GPU communication

### NVLink vs PCIe

- **NVLink** (within a node): 600 GB/s bidirectional → fast all-reduce for data parallelism
- **PCIe** (CPU-GPU, cross-node NIC): 32–64 GB/s → cross-node communication is the bottleneck

### Collective operations

| Operation | Description | Used for |
|-----------|-------------|---------|
| All-reduce | Sum across all GPUs | Gradient aggregation in data parallelism |
| All-gather | Concatenate across GPUs | Collecting sharded weights (FSDP) |
| Reduce-scatter | Partial reduce + scatter | FSDP gradient aggregation |
| All-to-all | Route data between GPUs | Expert parallelism in MoE |

**Ring all-reduce:** Each GPU sends to the next in a ring, passes forward received chunks, and accumulates. Total data transferred: $2(n-1)/n \approx 2$ messages per GPU, independent of number of GPUs.

See [parallelism.md](parallelism.md) for how these are used in training.
