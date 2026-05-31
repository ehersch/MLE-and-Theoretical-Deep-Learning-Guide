# Classifier-Free Guidance

An unconditional model generates random images. A conditional model generates images matching a description. **Classifier-free guidance** (CFG) is the simple trick that makes conditional generation work — and it's used in every major image and video generator today.

---

## The Goal: Conditional Generation

We want to condition generation on something — a text prompt, a class label, another image:

```
Unconditional:  model() → random image (could be anything)
Conditional:    model("a golden retriever playing fetch") → dog image

The condition c changes which distribution we sample from:
  p(x) → p(x | c)
```

The neural network just takes an extra input: the condition. A text prompt is encoded by a text encoder (usually CLIP or T5) and injected into the model via cross-attention.

---

## The Tradeoff: Diversity vs. Faithfulness

There's a fundamental tension in conditional generation:

```
Low conditioning strength → diverse but unfaithful
  "a golden retriever" → generates dogs, but maybe not a retriever,
                          maybe in unexpected poses, settings

High conditioning strength → faithful but less diverse
  "a golden retriever" → always generates a golden retriever,
                         but every image looks the same
```

CFG gives us a continuous knob to tune this tradeoff at inference time.

---

## Classifier Guidance (the original approach)

Before CFG, **classifier guidance** (Dhariwal & Nichol, 2021) used a separate classifier to steer generation:

```
At each denoising step:
  1. Compute the normal score (denoise direction)
  2. Compute the gradient of a noisy classifier: ∇_x log p(c | x_t)
  3. Add the classifier gradient to the score

Modified score = original score + w · ∇_x log p(c | x_t)
                                    ↑
                              "move toward higher probability of class c"
```

This worked but had a big problem: you need to train a **noise-robust classifier** (evaluated at noisy images $x_t$, not clean ones). That's a separate model requiring separate training. Complex and brittle.

---

## Classifier-Free Guidance: The Elegant Fix

**Ho & Salimans, 2022.** Key insight: we don't need a separate classifier. Train a single model for **both** conditional and unconditional generation, then implicitly compute the classifier gradient.

### Training

During training, randomly drop the conditioning signal (set $c = \emptyset$) with some probability (10–20%):

```python
def conditional_training_step(model, x1, c):
    t = torch.rand(x1.shape[0])
    x0 = torch.randn_like(x1)
    xt = (1-t) * x0 + t * x1        # (flow matching path)
    
    # Randomly drop condition (classifier-free guidance trick)
    mask = torch.rand(x1.shape[0]) < 0.1   # 10% chance
    c_input = [ci if not m else None for ci, m in zip(c, mask)]
    
    v_pred = model(xt, t, condition=c_input)
    return F.mse_loss(v_pred, x1 - x0)
```

The model learns both:
- $v_\theta(x_t, t, c)$ — conditional prediction
- $v_\theta(x_t, t, \emptyset)$ — unconditional prediction (same network, null condition)

### Generation: The CFG Formula

At inference, compute **both** predictions and extrapolate beyond the conditional one:

$$\hat{v}(x_t, t, c) = v_\theta(x_t, t, \emptyset) + w \cdot (v_\theta(x_t, t, c) - v_\theta(x_t, t, \emptyset))$$

where $w$ is the **guidance scale**.

```
Unconditional  +  w × (Conditional − Unconditional)
prediction            direction toward the condition
```

Rearranging:

$$\hat{v} = (1 + w) \cdot v_\theta(x_t, t, c) - w \cdot v_\theta(x_t, t, \emptyset)$$

So each inference step requires **two forward passes** (conditional + unconditional), then combines them.

---

## Intuition: Amplifying the Condition Signal

Think of it this way:

```
Unconditional velocity:  "move toward any natural image"
Conditional velocity:    "move toward a natural image matching 'golden retriever'"
Difference:              the direction that distinguishes dogs from everything else

CFG:  move in the conditional direction, but amplified by w
      → more aggressively toward "golden retriever-ness"
```

The guidance scale $w$ controls the amplification:

```
w = 0:   purely unconditional (ignores the prompt)
w = 1:   standard conditional (normal Bayes-optimal)
w = 7.5: typical for text-to-image (faithful, less diverse)
w = 20+: very faithful, may look unnatural ("over-saturated" aesthetics)
```

```
Low guidance (w=1):         High guidance (w=15):
┌─────────────────────┐     ┌─────────────────────┐
│ A dog. Could be any │     │ VERY CLEARLY a golden│
│ breed, setting.     │     │ retriever, perfectly │
│ Diverse, natural    │     │ matches the prompt.  │
│ looking.            │     │ Less natural.        │
└─────────────────────┘     └─────────────────────┘
```

---

## Negative Prompts

CFG enables a practical feature: **negative prompts**. Instead of using the unconditional model as the "repulsion" direction, use a different condition $c_{\text{neg}}$ (what you don't want):

$$\hat{v} = v_\theta(x_t, t, c_{\text{neg}}) + w \cdot (v_\theta(x_t, t, c_{\text{pos}}) - v_\theta(x_t, t, c_{\text{neg}}))$$

```
Positive prompt: "a golden retriever playing fetch on the beach"
Negative prompt: "blurry, dark, low quality, ugly, cartoon"

CFG:  move toward positive condition, away from negative condition
Result: high-quality realistic dog beach photo
```

Negative prompts are widely used in Stable Diffusion to improve image quality.

---

## CFG Trade-offs

**The cost:** two forward passes per step doubles compute. With 50 denoising steps, that's 100 model evaluations.

**Quality vs diversity:** guidance scale creates a Pareto frontier:

```
FID (diversity/quality) vs CLIP score (faithfulness):

         high CLIP score (faithful)
              ↑
         ●●●● ← w=7.5 sweet spot
        ●
       ●        ← w=1
FID  ──────────────────────────────────
(lower                               (higher
 = better quality)                    = worse)
```

---

## Guidance Distillation

Running two forward passes is expensive. Recent work distills CFG into a single-pass model.

**Classifier-free guidance distillation:** train a new model that directly outputs the guided prediction $\hat{v}$ in a single forward pass, matching the two-pass CFG outputs.

Used in **SDXL-Turbo, Flux-schnell**: single-step or few-step generation with quality comparable to many-step CFG. Much faster.

---

## Summary

```
Standard conditional training:  condition always provided
  → model learns p(x|c) but can't control faithfulness

CFG trick:
  Training:   randomly drop condition (10%) → model learns p(x|c) AND p(x)
  Inference:  extrapolate beyond conditional prediction
              ŷ = uncond + w·(cond - uncond)
  
  w=0:   unconditional
  w=7.5: typical
  w=∞:   would collapse to a single point (no diversity)

Cost: 2× compute per step (run model twice)
Benefit: continuous quality-diversity tradeoff at inference time,
         no retraining needed to change guidance scale
```

CFG is one of the most practically impactful ideas in the diffusion model literature. It's used in literally every major text-to-image model.
