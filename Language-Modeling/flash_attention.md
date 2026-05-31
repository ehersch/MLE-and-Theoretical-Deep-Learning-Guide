# FlashAttention

FlashAttention (Dao et al., 2022) is an IO-aware exact attention algorithm. It produces the same mathematical result as standard attention but reorders computation to minimize reads and writes to HBM, achieving dramatic speedups and memory savings.

---

## The problem with standard attention

Standard attention in pseudocode:

```python
S = Q @ K.T / sqrt(d_k)    # (T, T) — stored in HBM
A = softmax(S, dim=-1)      # (T, T) — stored in HBM
O = A @ V                   # (T, d) — stored in HBM
```

**Memory:** $O(T^2)$ for the attention matrix. For $T=8192$, $d=64$, BF16: $\sim 512$ MB per head.

**IO complexity:** The attention matrix is read and written to HBM multiple times:
1. Write $S$ after matmul
2. Read $S$ to compute softmax → write $A$
3. Read $A$ to compute $AV$

Each HBM read/write is slow (see [systems_and_hardware.md](systems_and_hardware.md)). For long sequences, attention is heavily memory-bandwidth-bound.

---

## The key insight: tiling

Instead of materializing the full $T \times T$ attention matrix, process it in **tiles**. Each tile fits in SRAM (fast). We compute the final output incrementally without storing intermediate $T \times T$ matrices.

The challenge: **softmax is non-local**. The denominator $\sum_j \exp(s_{ij})$ requires all $j$. We can't tile if we need to see all keys first.

**Solution: online softmax** (safe softmax with running statistics).

---

## Online Softmax

The standard numerically stable softmax computes:

$$\text{softmax}(x)_i = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}, \quad m = \max_j x_j$$

This requires two passes: one to find $m$, one to compute exp and normalize.

**Online softmax** merges these into one pass over streaming tiles:

Maintain running max $m$ and running sum $\ell$. When processing a new tile of scores $s_{\text{new}}$:

$$m_{\text{new}} = \max(m_{\text{old}}, \max(s_{\text{new}}))$$
$$\ell_{\text{new}} = \ell_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}}) + \sum_j \exp(s_{j,\text{new}} - m_{\text{new}})$$

The correction factor $\exp(m_{\text{old}} - m_{\text{new}})$ rescales the old sum to the new maximum.

We also update the output accumulator:

$$O_{\text{new}} = O_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}}) + \exp(s_{\text{new}} - m_{\text{new}}) V_{\text{new}}$$

At the end, divide: $O = O / \ell$.

---

## FlashAttention algorithm

```
for each query block Q_i:
    initialize: O_i = 0, ℓ_i = 0, m_i = -∞
    
    for each key-value block (K_j, V_j):
        S_ij = Q_i @ K_j.T / sqrt(d_k)    # in SRAM: (BLOCK_Q, BLOCK_K)
        apply causal mask to S_ij
        
        m_new = max(m_i, rowmax(S_ij))
        P_ij = exp(S_ij - m_new)           # in SRAM
        ℓ_new = exp(m_i - m_new) * ℓ_i + rowsum(P_ij)
        
        O_i = exp(m_i - m_new) * O_i + P_ij @ V_j
        m_i, ℓ_i = m_new, ℓ_new
    
    O_i = O_i / ℓ_i                        # final normalization
    write O_i to HBM
```

**IO complexity:** $O(T^2 d / M)$ HBM reads, where $M$ is SRAM size. Compare to standard attention: $O(T^2 + Td)$. For large $T$, this is a huge reduction.

**Memory complexity:** $O(T d)$ — we only store the output, never the full attention matrix.

---

## FlashAttention-2

FA2 (Dao, 2023) adds:
1. **Better parallelism over sequence length:** in FA1, the outer loop over Q blocks is parallelized but inner K/V loop is sequential per thread block. FA2 can also parallelize over K/V.
2. **Fewer non-matmul ops:** rescaling factors ($\exp(m_{\text{old}} - m_{\text{new}})$) are minimized.
3. **Work partitioning for MHA:** better load balancing across thread blocks.

Result: ~2× faster than FA1, ~9× faster than standard PyTorch attention on A100.

---

## FlashAttention-3

FA3 (2024) targets H100-specific features:
- **WGMMA (warp-group matrix multiply):** H100's new async tensor core instruction
- **TMA (Tensor Memory Accelerator):** async data movement from HBM to SRAM
- **Overlapping:** computation and data movement are pipelined

Achieves ~75% of H100's theoretical peak FLOP/s for attention.

---

## Backward pass

The backward pass for attention requires recomputing the softmax statistics ($m$, $\ell$) to reconstruct $P$ without having stored it. FA stores only $m$ and $\ell$ per row during the forward pass (not the full $T \times T$ attention matrix), then recomputes tiles as needed during backward.

This is the key insight that makes FA memory-efficient end-to-end: **recomputation is cheaper than storing $T^2$ values**.

---

## Using FlashAttention in practice

```python
# PyTorch >= 2.0: sdpa (scaled dot product attention) auto-selects FA
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

# Or directly via the flash_attn package
from flash_attn import flash_attn_func
out = flash_attn_func(Q, K, V, causal=True)
```

---

## Summary: why FlashAttention matters

| | Standard | FlashAttention 2 |
|---|---|---|
| Memory | $O(T^2)$ | $O(T)$ |
| HBM reads | $O(T^2)$ | $O(T^2 d / M)$ |
| Wall-clock speedup | 1× | 5–9× (A100, $T=4096$) |
| Max practical $T$ | ~4k | 100k+ |

FlashAttention made 100k+ context windows practical. Without it, fitting long sequences in memory requires model parallelism even for a single forward pass.
