# Discrete Diffusion Models

Diffusion and flow matching work naturally on continuous data (images, audio). But what about **discrete data** — text tokens, DNA sequences, protein residues? You can't add Gaussian noise to the integer "42" (the token for "cat").

Discrete diffusion defines analogous corruption and denoising processes for categorical data.

---

## Why Not Just Use Continuous Diffusion on Text?

The obvious attempt: embed text tokens into continuous vectors, run diffusion in embedding space, project back to tokens at the end.

```
tokens → embedding → [run diffusion] → embedding → round to nearest token
```

This works poorly in practice. The embedding space isn't designed for diffusion — the roundoff at the end loses information, and the geometry is wrong for denoising.

Discrete diffusion instead works **directly on the token indices**, defining a corruption process over categorical distributions.

---

## The Discrete Forward Process

Instead of adding Gaussian noise, we define a **corruption process** over token distributions. Three main approaches:

### Masking (Absorbing State)

Replace tokens with a special `[MASK]` token with some probability that increases over time:

```
t=0 (clean):   "the cat sat on the mat"
t=0.3:         "the [M] sat on the [M]"
t=0.7:         "the [M] [M] [M] [M] [M]"
t=1.0:         "[M] [M] [M] [M] [M] [M]"

Each token independently masked with probability t.
At t=1: fully masked (like starting from scratch)
```

This is the approach used in **Masked Diffusion Language Models (MDLMs)**.

### Uniform Noise

Replace each token with a uniformly random token from the vocabulary:

```
t=0 (clean):   "the cat sat on the mat"
t=0.5:         "the xyz sat on the qpr"  (random replacement)
t=1.0:         "jkw mno bcd efg hij"     (fully random tokens)
```

Less interpretable than masking but mathematically cleaner.

### Structured Noise (for DNA/proteins)

Use domain knowledge to define meaningful corruptions. For DNA: replace with biologically similar bases. For proteins: replace with amino acids with similar properties.

---

## Continuous-Time Markov Chains (CTMCs)

The formal mathematical framework for discrete diffusion. Instead of continuous noise levels, a **CTMC** defines transition rates between discrete states (tokens).

The key object is a **rate matrix** $Q$ where $Q_{ij}$ = rate of transitioning from token $i$ to token $j$.

```
For masking:
  Q_{i, MASK} = 1    (every token transitions to MASK at rate 1)
  Q_{MASK, MASK} = 0 (MASK is absorbing — never transitions away)
  Q_{i,j} = 0        (tokens don't transition to other tokens)

For uniform:
  Q_{i,j} = 1/(V-1) for j≠i   (uniform rate to any other token)
  Q_{i,i} = -1                  (diagonal ensures rows sum to 0)
```

Given $Q$, the marginal distribution at time $t$ has a closed-form expression (matrix exponential), just like the Gaussian noise schedule for continuous diffusion.

---

## Training: Denoising Corrupted Sequences

The training objective is essentially the same as continuous diffusion: given a corrupted sequence at "time" $t$, predict the original.

For the **masked** approach (most practical):

```python
def discrete_diffusion_training_step(model, tokens, t=None):
    B, L = tokens.shape   # batch, sequence length
    if t is None:
        t = torch.rand(B)   # random corruption level in [0,1]
    
    # Corrupt tokens: mask each with probability t
    mask_prob = t.unsqueeze(1).expand(B, L)           # (B, L)
    to_mask   = torch.bernoulli(mask_prob).bool()
    
    corrupted = tokens.clone()
    corrupted[to_mask] = MASK_TOKEN_ID                # apply masking
    
    # Model predicts original token at each masked position
    logits = model(corrupted, t)                      # (B, L, V)
    
    # Loss only on masked positions
    loss = F.cross_entropy(
        logits[to_mask],     # predicted logits at masked positions
        tokens[to_mask])     # true tokens at those positions
    return loss
```

The model sees: "here's a partially masked sequence, and the noise level $t$. What were the original tokens?"

---

## Sampling (Generation)

Generation runs the reverse: start fully masked, iteratively unmask tokens.

```
t=1.0 → fully masked:    [M] [M] [M] [M] [M] [M] [M]
t=0.7:                   "the [M] sat on [M] [M] [M]"
t=0.5:                   "the cat sat on [M] mat [M]"
t=0.3:                   "the cat sat on the mat [M]"
t=0.0 → complete:        "the cat sat on the mat ."
```

At each step:
1. Predict token probabilities at all masked positions
2. Sample tokens for some masked positions (or all, depending on the schedule)
3. Revealed tokens stay revealed (absorbing state)

```python
def discrete_diffusion_sample(model, length, n_steps=100):
    tokens = torch.full((1, length), MASK_TOKEN_ID)  # fully masked
    
    for step in range(n_steps):
        t = 1.0 - step / n_steps   # from 1.0 down to 0.0
        t_tensor = torch.tensor([t])
        
        # Predict original tokens
        logits = model(tokens, t_tensor)           # (1, L, V)
        probs  = logits.softmax(dim=-1)
        
        # Only sample at still-masked positions
        masked_positions = (tokens == MASK_TOKEN_ID)
        new_tokens = probs.argmax(dim=-1)           # or sample
        
        # Fraction of positions to unmask at this step
        unmask_fraction = 1.0 / (n_steps - step)
        confidence = probs.max(dim=-1).values       # (1, L)
        
        # Unmask the most confident masked positions
        n_to_unmask = int(masked_positions.sum() * unmask_fraction)
        confident_masked = confidence.masked_fill(~masked_positions, -1)
        top_positions = confident_masked.topk(n_to_unmask).indices
        tokens[0, top_positions] = new_tokens[0, top_positions]
    
    return tokens
```

---

## Masked Diffusion for Language Models

**The connection to BERT and masked language modeling (MLM):**

BERT already does something similar: mask tokens, predict them. Masked diffusion is essentially **BERT generalized to a continuous-time process with a diffusion framework**.

The key difference:
- BERT: mask 15%, predict all at once, one shot
- Masked diffusion: gradually increase masking from 0% to 100%, train to predict at every noise level, generate by iterative unmasking

**MDLM (Masked Diffusion Language Model, 2024):** a recent strong baseline. Achieves perplexity close to AR (autoregressive) models on text benchmarks while enabling **bidirectional** generation (not just left-to-right).

---

## Why Discrete Diffusion?

The main appeal over autoregressive (AR) language models:

```
AR (GPT-style):               Discrete Diffusion:
────────────────────────────────────────────────────
Generates left-to-right       Can generate any order
Token 1 depends on nothing    All tokens can depend on all others
Cannot "go back and revise"   Can unmask tokens that were wrong
Fast generation (parallel no) Parallel generation possible
Lossless compression: great   Can condition on arbitrary subsets
```

**Parallel generation:** the biggest win. Once you have a model that can fill in any masked positions, you can unmask all positions in **one step** (rough quality) or a few steps (good quality). This enables much faster generation than sequential AR.

**Current state (2025):** discrete diffusion hasn't yet matched GPT-4-class AR models on language quality, but the gap is narrowing and the architectural flexibility is compelling — especially for biology (proteins, DNA) where bidirectionality matters a lot.

---

## Discrete Flow Matching

Just as continuous diffusion generalizes to flow matching, there's a **discrete flow matching** formulation:

Instead of corrupting with masks, define a discrete "path" between data and noise distributions:

```
Data token:   cat   (one-hot: [0,0,...,1,...,0])
Noise token:  uniform over vocabulary

Interpolate: at time t, token is "cat" with prob (1-t), random with prob t
```

The training objective: given a token drawn from this mixture at time $t$, predict the data token.

Discrete flow matching (Gat et al., 2024) is cleaner than CTMC-based discrete diffusion and often works better in practice.
