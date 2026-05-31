# Inference

Inference is qualitatively different from training. The model is used autoregressively: generate one token, append it, generate the next. This creates unique memory and compute patterns.

---

## Autoregressive generation

The generation loop:

```python
def generate(model, prompt, max_new_tokens=256, temperature=1.0, top_p=0.9):
    tokens = tokenize(prompt)
    for _ in range(max_new_tokens):
        logits = model(tokens)[:, -1, :]      # only last token's logits
        logits = logits / temperature
        probs = top_p_sampling(softmax(logits), p=top_p)
        next_token = sample(probs)
        tokens = append(tokens, next_token)
        if next_token == EOS_TOKEN:
            break
    return tokens
```

**Prefill phase:** process the entire prompt in parallel → fast (like training).

**Decode phase:** generate one token at a time → slow (sequential, memory-bound).

The bottleneck is the decode phase. It's memory-bound because:
- Only 1 token is generated per step → tiny matmuls
- The full model must be read from HBM every step
- Arithmetic intensity $\approx$ batch_size (for small batch sizes << ridge point)

---

## KV Cache

In standard autoregressive generation without caching, we recompute K and V for all past tokens at every step. KV caching saves these:

**Without KV cache:** at step $t$, compute $Q, K, V$ for all $t$ tokens → $O(t)$ work per step → $O(T^2)$ total.

**With KV cache:** at step $t$, compute $Q, K, V$ only for the new token; retrieve cached $K, V$ for all previous tokens → $O(1)$ compute per step → $O(T)$ total.

```python
class KVCache:
    def __init__(self, max_seq_len, num_layers, num_heads, head_dim):
        self.k = torch.zeros(num_layers, max_seq_len, num_heads, head_dim)
        self.v = torch.zeros(num_layers, max_seq_len, num_heads, head_dim)
        self.pos = 0
    
    def update(self, layer, k_new, v_new):
        self.k[layer, self.pos] = k_new
        self.v[layer, self.pos] = v_new
        self.pos += 1
        return self.k[layer, :self.pos], self.v[layer, :self.pos]
```

**KV cache memory:** for each sequence:

$$\text{Memory} = 2 \times L \times T \times H \times d_h \times \text{dtype\_bytes}$$

For LLaMA-7B (L=32, H=32, $d_h$=128) at T=4096 in FP16: $2 \times 32 \times 4096 \times 32 \times 128 \times 2 \approx 2$ GB per sequence.

**At large batch sizes**, the KV cache dominates GPU memory — often more than the model weights. This limits how many requests can be processed concurrently.

---

## Sampling strategies

### Greedy decoding

$$x_t = \arg\max_v P(v \mid x_{<t})$$

Deterministic, often degenerates to repetitive outputs.

### Temperature sampling

$$P_T(v) = \text{softmax}(\text{logits} / T)$$

- $T < 1$: sharpen the distribution (more deterministic)
- $T > 1$: flatten (more random)
- $T \to 0$: greedy; $T \to \infty$: uniform

### Top-k sampling

Sample from only the $k$ most probable tokens. Truncates the long tail.

### Top-p (nucleus) sampling

Sample from the smallest set of tokens whose cumulative probability ≥ $p$:

```python
def top_p_sampling(probs, p=0.9):
    sorted_probs, sorted_idx = probs.sort(descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    # Find cutoff: first index where cumsum >= p
    cutoff = (cumsum >= p).float().argmax(dim=-1)
    # Zero out everything after cutoff
    sorted_probs[..., cutoff+1:] = 0
    sorted_probs /= sorted_probs.sum()
    return probs.scatter(-1, sorted_idx, sorted_probs)
```

Adapts to the distribution: when the model is confident (peaked), the nucleus is small.

### Min-p sampling

Remove tokens whose probability is less than $p \times \max(P)$. A multiplicative rather than additive threshold. Tends to outperform top-p empirically.

---

## Speculative Decoding

The key insight: the large model (target) is slow; a small draft model is fast. If the draft is usually right, we can amortize target model calls.

**Algorithm:**
1. Draft model generates $k$ tokens speculatively
2. Target model processes all $k$ tokens in **one parallel forward pass**
3. Accept/reject each draft token via a correction scheme that preserves the target distribution

**Acceptance/rejection:**

Let $p(x)$ = target probability, $q(x)$ = draft probability for token $x$.

For each draft token $x_i$:
- Accept with probability $\min(1, p(x_i)/q(x_i))$
- If rejected: sample from corrected distribution $\text{norm}(\max(0, p - q))$

This guarantees the output distribution is **exactly** the target model's distribution — no quality degradation.

**Speedup:** If draft model accepts $\alpha$ fraction of tokens on average, and target model processes $k+1$ tokens in one forward pass:

$$\text{Speedup} = \frac{k\alpha + 1}{(k+1) / \text{draft\_speed\_ratio}}$$

Typical: 2–3× wall-clock speedup with a good draft model.

```python
def speculative_decode(target, draft, prompt, k=5):
    tokens = prompt
    while not done:
        # Draft model generates k tokens
        draft_tokens, draft_probs = [], []
        for _ in range(k):
            p = draft.forward(tokens)
            x = sample(p)
            draft_tokens.append(x)
            draft_probs.append(p[x])
            tokens = append(tokens, x)
        
        # Target model scores all k draft tokens in one pass
        target_logits = target.forward(tokens[:-k+1:])  # full context
        target_probs = [softmax(target_logits[i])[draft_tokens[i]] for i in range(k)]
        
        # Accept/reject
        accepted = 0
        for i in range(k):
            r = random.uniform(0, 1)
            if r < target_probs[i] / draft_probs[i]:
                accepted += 1
            else:
                # Resample from corrected distribution and break
                resample_and_break(target_logits[i], draft_probs[i])
                break
```

**Variants:**
- **Medusa:** multiple draft heads on the target model itself (no separate draft model)
- **EAGLE:** lightweight draft model that uses target's hidden states
- **Self-speculative decoding:** skip some layers to create the draft

---

## Continuous Batching

Naively, a serving system waits for all sequences in a batch to finish before starting new ones. But sequences finish at different times → GPU sits idle waiting for the longest sequence.

**Continuous batching (iteration-level scheduling):** the scheduler inserts new requests into the batch at every token step. Once a sequence finishes (EOS), its slot is immediately reused.

```
Step 1: [req A (pos 5), req B (pos 3), req C (pos 1)]
Step 2: [req A (pos 6), req B (pos 4), req D (pos 1)]  ← req C finished, req D inserted
```

This dramatically improves GPU utilization. Used in vLLM, TGI, TensorRT-LLM.

---

## PagedAttention (vLLM)

KV cache memory must be pre-allocated for each sequence's maximum length. But most sequences don't reach max length — wasted memory.

**PagedAttention** (Kwon et al., 2023) treats KV cache like virtual memory paging:
- KV cache is divided into fixed-size blocks (pages)
- Each sequence's KV cache is stored in non-contiguous blocks
- A block table maps logical KV positions to physical memory blocks
- Blocks can be shared between sequences (useful for beam search / shared prefixes)

Result: near-zero fragmentation and memory waste. Enables much higher throughput (2–4× more concurrent requests).

---

## Quantization at inference time

See [quantization.md](quantization.md) for details. Key point: quantization reduces model weight size, which:
- Reduces HBM reads per step → higher arithmetic intensity → faster decode
- Fits larger models on fewer GPUs
- INT8 weights, FP16 compute: effectively 2× faster memory-bound throughput

---

## Inference serving: putting it together

Production LLM inference (e.g., vLLM):

```
Request queue → Continuous batcher → GPU cluster
                                          ↓
                     PagedAttention KV cache management
                                          ↓
                     Tensor-parallel model execution (TP=4 or 8)
                                          ↓
                     Speculative decoding (optional)
                                          ↓
                     Token streaming back to client
```

Key metrics:
- **Throughput:** tokens/second across all users
- **Latency:** time to first token (TTFT) and time per output token (TPOT)
- **GPU utilization:** fraction of time tensor cores are active
