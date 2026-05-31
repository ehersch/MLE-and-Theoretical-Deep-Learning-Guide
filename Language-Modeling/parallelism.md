# Parallelism in LLM Training

A 7B model needs ~84 GB just for weights + optimizer states. A 70B model needs ~840 GB. No single GPU has that much memory. We distribute across GPUs using multiple parallelism strategies.

---

## The four types of parallelism

| Type | Splits | Communication | Use for |
|------|--------|---------------|---------|
| Data | Batch | All-reduce gradients | Scale throughput |
| Tensor | Weight matrices | All-reduce activations | Large layers |
| Pipeline | Layers | Point-to-point activations | Very deep models |
| Sequence | Sequence length | All-gather/reduce | Long context |

These are often combined: a 4D parallelism (TP × PP × DP × SP).

---

## Data Parallelism (DP)

The simplest strategy: each GPU holds a **full copy** of the model. Each GPU processes a different micro-batch. After the backward pass, gradients are **all-reduced** across GPUs and each GPU updates its own weights.

```
GPU 0: model copy, batch[0] → grad[0] ──┐
GPU 1: model copy, batch[1] → grad[1] ──┤─ all-reduce → avg grad → update
GPU 2: model copy, batch[2] → grad[2] ──┘
```

**Limitation:** requires entire model + optimizer states to fit on one GPU. Fine for small models, not for 70B+.

**DDP (Distributed Data Parallel)** in PyTorch: gradient buckets are communicated as they become ready during backward, overlapping communication with computation.

```python
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[rank])
# Gradients are automatically averaged across ranks after backward()
```

---

## ZeRO: Zero Redundancy Optimizer

ZeRO (Rajbhandari et al., 2020) eliminates the redundancy of storing full optimizer states on every GPU. Three stages:

### ZeRO Stage 1: Partition optimizer states

Each GPU stores $1/N$ of the optimizer states (Adam m, v). After gradient all-reduce, each GPU updates its shard of parameters.

- Memory per GPU (optimizer states): $12N/\text{num\_gpus}$ → 12× reduction

### ZeRO Stage 2: + Partition gradients

Each GPU also stores only $1/N$ of gradients. Reduce-scatter instead of all-reduce.

### ZeRO Stage 3: + Partition parameters (FSDP)

Each GPU stores only $1/N$ of parameters too. Before a layer's forward pass, **all-gather** the full parameter shard; after, discard it.

```
Forward:  all-gather params → compute → discard params
Backward: all-gather params → compute grad → reduce-scatter grads → discard params
```

Memory per GPU: $\approx 16N / \text{num\_gpus}$ bytes (total), vs $16N$ in naive DP.

**PyTorch FSDP (Fully Sharded Data Parallel)** implements ZeRO-3:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, auto_wrap_policy=transformer_auto_wrap_policy)
```

**Communication cost:** ZeRO-3 adds all-gathers for forward and backward — twice the communication of ZeRO-1. Worth it for large models where memory is the constraint.

---

## Tensor Parallelism (TP)

Split individual weight matrices across GPUs. Used within a node (via fast NVLink).

### Column-parallel linear

Split $W \in \mathbb{R}^{d \times h}$ column-wise across $N$ GPUs. Each GPU holds $W_i \in \mathbb{R}^{d \times h/N}$:

$$Y_i = X W_i$$

Gather $Y = [Y_1, \ldots, Y_N]$ via all-gather.

### Row-parallel linear

Split $W$ row-wise. Each GPU holds $W_i \in \mathbb{R}^{d/N \times h}$ and its input shard $X_i$:

$$Y = \sum_i X_i W_i$$

Sum via all-reduce.

### Attention tensor parallelism

Split attention heads across GPUs. Each GPU handles $h/N$ heads independently for Q, K, V projections. The output projection is row-parallel.

```
GPU 0: heads 0..7   → partial output
GPU 1: heads 8..15  → partial output  →  all-reduce  →  full output
GPU 2: heads 16..23 → partial output
GPU 3: heads 24..31 → partial output
```

**Communication:** one all-reduce per transformer layer (for attention) + one per FFN = 2 all-reduces per layer. Very efficient on NVLink, expensive across nodes.

---

## Pipeline Parallelism (PP)

Split the model **layer-by-layer** across GPUs. Each GPU holds a contiguous chunk of layers.

```
GPU 0: layers 0-7   → activations
GPU 1: layers 8-15  → activations
GPU 2: layers 16-23 → activations
GPU 3: layers 24-31 → logits
```

**Naive issue:** while GPU 1 is processing, GPU 0 is idle (pipeline bubble). Bubble fraction $= \frac{p-1}{p + m - 1}$ where $p$ = pipeline stages, $m$ = micro-batches.

**1F1B schedule (One Forward, One Backward):** interleave micro-batches to fill the pipeline. With $m \gg p$, bubble fraction → 0.

**Communication:** only send activations at stage boundaries (small). Good for multi-node training where cross-node bandwidth is limited.

---

## Sequence Parallelism (SP)

For very long sequences, the attention matrix $T \times T$ doesn't fit on one GPU. Split along the sequence dimension.

**Ring attention:** Each GPU holds $T/N$ tokens. During attention, GPUs pass K/V blocks in a ring pattern, accumulating partial attention outputs. Each GPU sees all K/V but only computes its own Q's attention.

```python
# Conceptually: ring all-to-all of K, V
for step in range(num_gpus):
    k_block = recv_from_prev()
    partial_attn += attention(Q_local, k_block, v_block)
    send_to_next(k_block)
```

Used for 100k+ context window training.

---

## Gradient accumulation

When per-GPU batch size must be small (memory), simulate a large batch by accumulating gradients over multiple micro-batches before an optimizer step:

```python
optimizer.zero_grad()
for i, micro_batch in enumerate(micro_batches):
    loss = model(micro_batch) / num_micro_batches
    loss.backward()  # gradients accumulate
optimizer.step()
```

**Effective batch size** = per-GPU batch size × gradient accumulation steps × num GPUs.

---

## 3D/4D parallelism in practice

Large-scale training (e.g., GPT-4, LLaMA-3 405B) combines:

- **TP** within a node (8-way, using NVLink)
- **PP** across nodes (to minimize cross-node traffic; only activations at boundaries)
- **DP** as outer loop (many copies of the TP+PP group)
- **SP** for long context runs

Example: 1000 A100s (125 nodes × 8 GPUs):
- TP=8 within each node
- PP=8 across 8 nodes
- DP=125/8 ≈ 16 data-parallel replicas

**Memory-compute tradeoff:** tensor parallelism reduces memory but requires fast interconnect. Pipeline parallelism is cheaper on communication but has bubble overhead and complex scheduling.

---

## Activation checkpointing (gradient checkpointing)

During backward, we need the activations from the forward pass. Storing all of them is expensive:

- Full activation storage: $O(Ld)$ per sequence per layer = GBs
- **Checkpointing:** only store activations at certain layer boundaries; recompute intermediate activations during backward

Cost: ~33% extra compute in exchange for dramatically lower memory. Standard for large-model training.

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.layer1, x)  # activations not stored
    x = checkpoint(self.layer2, x)
    return x
```
