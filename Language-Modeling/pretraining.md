# Pretraining

Pretraining is the dominant cost in LLM development: a single run of a frontier model consumes millions of GPU-hours. The goal is to train a model that has compressed statistical structure of language and world knowledge from massive text corpora.

---

## Pretraining objectives

### Causal Language Modeling (CLM)

The standard objective for decoder-only models. Predict the next token given all previous tokens:

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})$$

Simple, scalable, and all tokens provide gradient signal simultaneously (unlike RNNs which must unroll sequentially).

### Masked Language Modeling (MLM)

Used by BERT. Randomly mask 15% of tokens and predict the masked positions:

$$\mathcal{L}_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log P_\theta(x_t \mid x_{\setminus \mathcal{M}})$$

Bidirectional context → better representations. But not directly usable for generation. BERT uses a 80/10/10 masking strategy: 80% `[MASK]`, 10% random token, 10% original (to keep the model calibrated).

### Span Corruption (T5)

Mask contiguous spans of tokens (not individual tokens) and predict them:

```
Input:  "The [X] sat on [Y] mat"
Target: "[X] cat [Y] the"
```

Forces the model to generate multi-token continuations, bridging MLM and CLM.

### Next Sentence Prediction (NSP) / Sentence Order Prediction

BERT's auxiliary objective (since largely abandoned — RoBERTa showed it doesn't help).

---

## Training data

Scale and quality of data determines model capability as much as architecture.

### Common datasets

| Dataset | Size | Source |
|---------|------|--------|
| Common Crawl (C4) | ~750GB text | Web crawl |
| The Pile | 825GB | 22 curated sources |
| RedPajama | 1.2T tokens | Open reproduction of LLaMA data |
| FineWeb | 15T tokens | CC with careful filtering |
| DCLM | 4T tokens | Carefully filtered CC |
| Dolma | 3T tokens | OLMo training data |

### Data pipeline

```
Raw web crawl (petabytes)
    ↓ URL/domain filtering
    ↓ Language identification (fastText)
    ↓ Quality filtering (heuristics: length, punctuation ratio, etc.)
    ↓ Deduplication (MinHash LSH or exact substring match)
    ↓ Content filtering (NSFW, PII)
    ↓ Tokenization
    ↓ Shuffling and packing into fixed-length sequences
```

**Deduplication matters enormously.** Common Crawl contains vast amounts of near-duplicate content. Models trained on deduplicated data generalize better and memorize less.

**Packing:** Concatenate documents with `<EOS>` separators to fill $T_{\max}$-length sequences, maximizing GPU utilization. Attention should be masked to not cross document boundaries (though this is sometimes ignored in practice).

### Data mixing and weighting

Different data sources should be weighted during training:

$$\mathcal{L} = \sum_s w_s \mathcal{L}_s$$

LLaMA-1 upweights code, math, and Wikipedia relative to their raw proportion. Too much low-quality web text degrades performance; too much high-quality narrow data limits diversity.

---

## Hyperparameters and training stability

### Learning rate schedule

The standard is a warmup followed by cosine decay:

$$\eta_t = \begin{cases}
\eta_{\max} \cdot t / T_{\text{warmup}} & t < T_{\text{warmup}} \\
\eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi \cdot \frac{t - T_{\text{warmup}}}{T - T_{\text{warmup}}})) & \text{otherwise}
\end{cases}$$

Warmup prevents large gradient updates before the model has meaningful representations.

### Optimizer: AdamW

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_{t-1}$$

The last term is **weight decay** (L2 regularization decoupled from the adaptive LR). Typical: $\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$, $\lambda=0.1$.

**Why not SGD?** LLM loss landscapes are very non-convex. Adaptive methods like Adam handle different parameter scales automatically. The second moment $v_t$ acts as a per-parameter learning rate normalizer.

### Gradient clipping

Large gradients (loss spikes) can destabilize training. Clip by global norm:

$$g \leftarrow g \cdot \min\!\left(1, \frac{c}{\|g\|}\right)$$

Typical threshold: $c = 1.0$.

### Loss spikes and instability

At large scale, training often exhibits sudden loss spikes followed by recovery. Mitigations:
- Gradient clipping
- Lower LR peak
- QK-Norm (normalize Q and K before attention, preventing attention logit explosion)
- z-loss: auxiliary loss penalizing large logit magnitudes

---

## Mixed precision training

Training in FP32 is wasteful. Modern training uses:

- **BF16** for forward pass and activations (8-bit exponent, same as FP32 → no overflow)
- **FP32** for optimizer states (master weights, Adam moments)
- **FP16 + loss scaling** alternatively (but BF16 is preferred on A100/H100)

Memory breakdown for a 7B model in BF16 training:
- Model weights: $7\text{B} \times 2$ bytes = 14 GB
- Gradients: 14 GB
- Adam m, v states in FP32: $7\text{B} \times 8$ bytes = 56 GB
- **Total: ~84 GB** (one A100 = 80 GB → need parallelism)

---

## Context length during pretraining

Training at long context is expensive ($O(T^2)$ attention). Common practice:
1. Pretrain at short context (e.g., 4k tokens)
2. Continue training at longer context (e.g., 128k tokens) on a subset of data with YaRN or NTK RoPE scaling

This saves compute while achieving long-context capability.

---

## Checkpointing and resumption

Pretraining runs for weeks or months. Checkpoints save:
- Model weights
- Optimizer states (critical for resumption)
- RNG states (for reproducibility)
- Data loader state (which documents have been seen)

Without data loader state, resuming shuffles data differently, effectively re-seeing some data and missing other data — subtly corrupting the training distribution.

---

## What does a pretrained model know?

A well-pretrained model has learned:
- Syntax and grammar (emergent from prediction)
- World facts embedded in weights
- Reasoning patterns (chain-of-thought can be prompted)
- Code understanding and generation
- Multiple languages

But it is NOT yet an assistant. It will continue any prompt — including harmful ones — because it has no objective to be helpful. Post-training is required (see [posttraining.md](posttraining.md)).
