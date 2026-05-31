# Attention Variants and Architecture Alternatives

Standard multi-head attention (MHA) is computationally expensive: $O(T^2 d)$ time and $O(T^2)$ memory. This drives a rich space of alternatives, as well as variants that reduce KV cache memory.

---

## KV cache reduction: MQA, GQA

The KV cache is a major bottleneck at inference. The key insight: queries need to be unique per head, but keys and values can be shared.

### Multi-Query Attention (MQA)

All query heads share a single K and V head:

$$Q_1, \ldots, Q_h \in \mathbb{R}^{T \times d_k}, \quad K \in \mathbb{R}^{T \times d_k}, \quad V \in \mathbb{R}^{T \times d_v}$$

- KV cache reduced by $h\times$ (e.g., 32× for LLaMA-70B)
- Quality degrades somewhat vs MHA
- Used by PaLM, Falcon

### Grouped-Query Attention (GQA)

A middle ground: group query heads, share K/V within each group. With $G$ groups and $h$ heads, each group has $h/G$ query heads sharing one K/V head.

$$\text{MHA} = \text{GQA with } G=h, \quad \text{MQA} = \text{GQA with } G=1$$

- LLaMA-2 70B, LLaMA-3, Mistral use GQA with $G=8$
- Near-MHA quality, near-MQA inference speed

```python
# GQA attention (simplified)
def gqa(Q, K, V, num_heads, num_kv_heads):
    # Q: (B, num_heads, T, d_k)
    # K, V: (B, num_kv_heads, T, d_k)
    groups = num_heads // num_kv_heads
    # Repeat K, V for each group
    K = K.repeat_interleave(groups, dim=1)  # (B, num_heads, T, d_k)
    V = V.repeat_interleave(groups, dim=1)
    return standard_attention(Q, K, V)
```

---

## Sparse Attention

For very long sequences, computing all $O(T^2)$ attention weights is infeasible. Sparse attention restricts which positions can attend to which.

### Sliding Window Attention (Longformer, Mistral)

Each token attends to a local window of size $w$:

$$A_{ij} \neq 0 \text{ only if } |i - j| \leq w/2$$

Complexity: $O(Tw)$. Can add global tokens (like `[CLS]`) that attend everywhere.

**Mistral-7B** uses sliding window attention with $w=4096$ over a 32k context. Because information propagates through layers, the effective receptive field grows with depth.

### Strided / Dilated Attention

Alternate between local and strided patterns to cover both local and global context without full $O(T^2)$.

### BigBird

Combines: random attention + local window + global tokens. Provably Turing-complete and handles sequences up to 4k+ efficiently.

---

## Linear Attention

Replace the softmax with a kernel approximation to achieve $O(T)$ complexity.

General form of attention:

$$\text{Attn}(q_i, K, V) = \frac{\sum_j k(q_i, k_j) v_j}{\sum_j k(q_i, k_j)}$$

For softmax, $k(q, k) = \exp(q \cdot k / \sqrt{d})$, which requires computing all $j$ before normalizing.

**Performer (FAVOR+):** Approximate $\exp(q \cdot k)$ with a random feature map $\phi$:

$$\exp(q \cdot k) \approx \phi(q)^\top \phi(k)$$

Then by associativity:

$$\sum_j \phi(q)^\top \phi(k_j) v_j = \phi(q)^\top \underbrace{\left(\sum_j \phi(k_j) v_j^\top\right)}_{S}$$

$S$ can be computed incrementally, giving $O(T)$ attention.

**Practical issue:** Linear attention approximations often underperform softmax attention in practice, especially for tasks requiring precise retrieval.

---

## State Space Models (SSMs) and Mamba

SSMs model sequences as a continuous dynamical system discretized to:

$$h_t = Ah_{t-1} + Bx_t, \quad y_t = Ch_t$$

where $A, B, C$ are learned matrices and $h_t$ is a hidden state.

This is equivalent to a convolution during training (parallel scan), and a recurrence during inference — giving $O(T)$ training with $O(1)$ per-step inference cost.

**Mamba** adds **selective state spaces**: the $B, C$ matrices become input-dependent ($B_t = f(x_t)$), allowing the model to selectively remember or forget. This closes much of the quality gap with transformers.

```
Mamba block:
    x → Linear → SSM (selective) → SiLU → output
              ↗
    x → Linear (gate)
```

**Hybrid models** (e.g., Jamba, Zamba) interleave Mamba and transformer layers to get the best of both.

---

## Mixture of Experts (MoE)

Increase model capacity without proportionally increasing compute by routing each token to a subset of FFN "experts."

$$\text{MoE}(x) = \sum_{i=1}^k g_i(x) \cdot \text{FFN}_i(x)$$

**Router:** a linear layer produces logits over $N$ experts; top-$k$ experts (usually $k=2$) are selected via softmax.

$$g(x) = \text{top-}k\text{-softmax}(xW_r)$$

**Key numbers (Mixtral-8×7B):**
- 8 experts, 2 active per token
- Each expert has 7B-equivalent FFN
- Total params: ~47B, but only ~13B active per forward pass
- Training cost ≈ 13B dense model; quality ≈ 70B dense model

**Load balancing:** Without auxiliary losses, the router collapses to always using the same expert. An auxiliary load-balancing loss penalizes uneven expert usage:

$$\mathcal{L}_{\text{aux}} = \alpha \sum_i f_i \cdot P_i$$

where $f_i$ = fraction of tokens routed to expert $i$, $P_i$ = mean routing probability to expert $i$.

**Expert parallelism:** In distributed training, different experts live on different devices. Tokens are dispatched via all-to-all communication.

---

## Summary table

| Variant | Complexity | Key benefit | Used by |
|---------|-----------|-------------|---------|
| MHA | $O(T^2 d)$ | Full expressiveness | GPT-2, BERT |
| MQA | $O(T^2 d/h)$ KV | Tiny KV cache | PaLM, Falcon |
| GQA | $O(T^2 d/G)$ KV | Balanced quality/speed | LLaMA-2/3, Mistral |
| Sliding window | $O(Twd)$ | Long context | Mistral, Longformer |
| Linear (Performer) | $O(Td^2)$ | Linear in $T$ | Research |
| Mamba (SSM) | $O(Td^2)$ | $O(1)$ inference | Mamba, Jamba |
| MoE | $O(T^2 d \cdot k/N)$ active | Capacity without compute | Mixtral, GPT-4 |
