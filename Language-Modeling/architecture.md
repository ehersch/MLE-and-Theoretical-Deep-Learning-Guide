# Transformer Architecture

The transformer (Vaswani et al., 2017) replaced recurrence with self-attention, enabling full parallelism during training and better long-range dependency modeling. Modern LLMs are decoder-only transformers with various architectural refinements.

---

## The language modeling objective

A language model assigns probability to sequences. The autoregressive factorization:

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, \ldots, x_{t-1})$$

We train to minimize negative log-likelihood (cross-entropy loss):

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_{<t})$$

This is also called the **causal language modeling (CLM)** objective.

---

## High-level architecture (decoder-only)

```
Input tokens → Token Embedding + Positional Encoding
    ↓
┌─────────────────────────────┐
│  Transformer Block × L      │
│  ┌─────────────────────┐    │
│  │ RMSNorm              │    │
│  │ Causal Self-Attention│    │
│  │ Residual add         │    │
│  │ RMSNorm              │    │
│  │ FFN (SwiGLU)         │    │
│  │ Residual add         │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
    ↓
Final LayerNorm → Linear → Softmax over vocab
```

---

## Self-Attention

Given input $X \in \mathbb{R}^{T \times d}$, we project to queries, keys, values:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$.

Attention scores and output:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

The $\sqrt{d_k}$ scaling prevents softmax saturation when $d_k$ is large (dot products grow in magnitude with dimension).

**Causal masking:** For autoregressive generation, we mask out future positions:

$$A_{ij} = \begin{cases} \frac{q_i \cdot k_j}{\sqrt{d_k}} & j \leq i \\ -\infty & j > i \end{cases}$$

After softmax, $-\infty$ entries become 0, so position $i$ only attends to positions $\leq i$.

```python
import torch
import torch.nn.functional as F

def causal_self_attention(Q, K, V):
    T, d_k = Q.shape[-2], Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d_k**0.5  # (B, H, T, T)
    mask = torch.triu(torch.ones(T, T, device=Q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return attn @ V
```

---

## Multi-Head Attention (MHA)

Instead of one attention function, run $h$ parallel heads on lower-dimensional projections:

$$\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)$$

$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

Each head can attend to different aspects of the sequence. Total parameters: $4d^2$ (same as single-head when $d_k = d/h$).

---

## Feed-Forward Network (FFN)

Each position is processed independently through a 2-layer MLP:

$$\text{FFN}(x) = \text{activation}(xW_1 + b_1) W_2 + b_2$$

**SwiGLU** (used in LLaMA, PaLM): a gated variant that empirically works better:

$$\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_2) W_3$$

$$\text{Swish}(x) = x \cdot \sigma(x)$$

The intermediate dimension is typically $4d$ (vanilla) or $\frac{8d}{3}$ (SwiGLU, to keep parameter count similar).

---

## Layer Normalization

Original transformer used Post-LN (normalize after residual). Modern LLMs use **Pre-LN** (normalize before each sub-layer) for training stability:

$$\text{Pre-LN:} \quad x \leftarrow x + \text{sublayer}(\text{Norm}(x))$$

**RMSNorm** (used in LLaMA, Gemma) drops the mean-centering of LayerNorm, keeping only the RMS rescaling:

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2}$$

Cheaper to compute, works just as well empirically.

---

## Positional Encodings

Attention is permutation-invariant without positional information. We need to inject position.

### Sinusoidal (original)

$$\text{PE}_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$$
$$\text{PE}_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Added to token embeddings. Generalizes to longer sequences than seen during training (in theory), but in practice models degrade.

### Learned absolute (GPT-2)

A learned embedding $E_{\text{pos}} \in \mathbb{R}^{T_{\max} \times d}$. Simple, works well within context length, but doesn't extrapolate.

### Rotary Position Embedding (RoPE)

Used by LLaMA, GPT-NeoX. Rather than adding position to the embedding, **rotate** the query and key vectors by an angle proportional to position before computing dot products:

$$q_m \cdot k_n = \text{Re}\left[ (R_m q_m)^* (R_n k_n) \right]$$

where $R_m$ is a block-diagonal rotation matrix. The dot product ends up depending only on the **relative** position $m - n$, not absolute positions.

Key benefit: relative position awareness; better length generalization (with NTK/YaRN extensions).

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    # cos, sin: (T, d_head/2) broadcast over batch/heads
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
```

### ALiBi (Attention with Linear Biases)

Add a learned linear penalty to attention logits based on distance:

$$A_{ij} \leftarrow A_{ij} - m \cdot |i - j|$$

where $m$ is a per-head slope. No position added to embeddings. Extrapolates gracefully beyond training context length.

---

## Model scale: counting parameters

For a transformer with vocab size $V$, $L$ layers, hidden dim $d$, FFN intermediate $d_{\text{ff}}$, heads $h$:

| Component | Parameters |
|-----------|-----------|
| Token embedding | $V \cdot d$ |
| Per layer: attention | $4d^2$ (Q, K, V, O projections) |
| Per layer: FFN (SwiGLU) | $3 \cdot d \cdot d_{\text{ff}}$ |
| Per layer: norms | $2d$ |
| Output head (often tied) | $V \cdot d$ |

**Total ≈** $2Vd + L(4d^2 + 3dd_{\text{ff}})$ (ignoring small terms)

For LLaMA-7B: $L=32$, $d=4096$, $d_{\text{ff}}=11008$, $V=32000$ → ~7B parameters. ✓

---

## Encoder vs Decoder vs Encoder-Decoder

| Architecture | Examples | Use case |
|---|---|---|
| Encoder-only | BERT, RoBERTa | Classification, NER, embeddings |
| Decoder-only | GPT, LLaMA, Gemma | Generation, instruction following |
| Encoder-Decoder | T5, BART | Translation, summarization |

Modern LLMs are almost exclusively **decoder-only**. The bidirectional encoder is useful for representation tasks but not for generation. T5 showed encoder-decoder can be unified under text-to-text, but decoder-only scaling has proven most effective.

---

## KV cache and inference shape

During generation, at each step we only compute attention for the **new** token. But we need $K, V$ for all past tokens. We cache them:

- Cache size per layer: $2 \times T \times d$ (K and V, $T$ tokens so far)
- Total cache: $2LTd$ floats → for LLaMA-7B, $T=2048$: ~1GB in fp16

This is why long-context models are memory-hungry at inference. (See [inference.md](inference.md) for full treatment.)
