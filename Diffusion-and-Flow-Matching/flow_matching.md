# Flow Matching

Flow matching is arguably the cleanest way to think about generative models. The idea: define a smooth path between noise and data, then train a neural network to follow it. No probabilistic tricks required.

---

## The Setup

We want to transport samples from a simple source distribution $p_0 = \mathcal{N}(0, I)$ (noise) to a complex target distribution $p_1 = p_{\text{data}}$ (real images).

Imagine each data point $x_1 \sim p_{\text{data}}$ gets paired with a noise sample $x_0 \sim \mathcal{N}(0,I)$. We draw a **path** between them:

```
Noise        t=0.25       t=0.5        t=0.75       Data
  x₀   ─────────────────────────────────────────→   x₁

The path: x_t = (1-t)·x₀ + t·x₁   (straight line!)

t=0: x_t = x₀  (pure noise)
t=1: x_t = x₁  (pure data)
t=0.5: x_t = average of noise and data
```

The **velocity** along this path at time $t$ is just the derivative:

$$\frac{d x_t}{dt} = x_1 - x_0$$

That's it. The velocity from noise to a specific data point is just their difference.

---

## The Problem: We Don't Have Pairs

In practice, we don't have a canonical pairing of noise samples to data samples. We just have data $\{x_1^{(i)}\}$ and can sample noise $x_0 \sim \mathcal{N}(0, I)$.

**The marginal vector field:** if we average the conditional velocity $(x_1 - x_0)$ over all possible pairings, we get the **marginal velocity field** $v(x_t, t)$ — the average direction to move from $x_t$ at time $t$:

```
At a given point x_t in space:

Multiple data paths pass through here (different x₁ values):
        ↗ (heading toward data point A)
  x_t ──→ (heading toward data point B)
        ↘ (heading toward data point C)

Marginal velocity = weighted average of all these directions
                  = expected (x₁ - x₀) given x_t
```

Importantly: **we never need to compute this marginal velocity explicitly**. The flow matching training objective avoids it entirely.

---

## The Flow Matching Training Objective

**Conditional flow matching** trains a neural network $v_\theta(x_t, t)$ to predict the conditional velocity $(x_1 - x_0)$ for a specific pair $(x_0, x_1)$:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1}\left[\| v_\theta(x_t, t) - (x_1 - x_0) \|^2\right]$$

where:
- $t \sim \text{Uniform}(0, 1)$ — random time
- $x_0 \sim \mathcal{N}(0, I)$ — noise sample
- $x_1 \sim p_{\text{data}}$ — real data sample
- $x_t = (1-t) x_0 + t x_1$ — point on the path at time $t$

```python
def flow_matching_loss(model, x1, t=None):
    B = x1.shape[0]
    if t is None:
        t = torch.rand(B)                         # random time in [0,1]
    
    x0 = torch.randn_like(x1)                     # sample noise
    
    # Interpolate between noise and data
    t_expand = t.view(B, *([1]*(x1.ndim-1)))
    xt = (1 - t_expand) * x0 + t_expand * x1     # point on path
    
    # Target velocity: direction from noise to data
    target_velocity = x1 - x0
    
    # Predict velocity
    predicted_velocity = model(xt, t)
    
    return F.mse_loss(predicted_velocity, target_velocity)
```

This is remarkably clean: the loss says "given a noisy image at time $t$, predict which direction to move to get to the real image." No complex probability theory, no ELBO, no score functions — just regression on velocity.

---

## Sampling (Generation)

After training, generate images by integrating the ODE:

```python
def flow_matching_sample(model, shape, n_steps=100):
    x = torch.randn(shape)   # start from pure noise
    
    for i in range(n_steps):
        t = torch.full((shape[0],), i / n_steps)
        v = model(x, t)
        x = x + (1.0 / n_steps) * v   # Euler step
    
    return x
```

```
t=0    t=0.1   t=0.2   t=0.3  ...  t=0.9   t=1.0
noise   →       →       →           →      image
 ●                                           ●
  ╰──────────────────────────────────────────╯
               ODE trajectory
```

With enough steps this gives high-quality samples. With just 10-20 steps (using better ODE solvers), quality is still excellent — a key advantage over DDPM which needs 1000 steps.

---

## Rectified Flow: Straight Paths

**Rectified flow** (Liu et al., 2022) makes the paths as straight as possible. Straight paths mean:
- Fewer integration steps needed (straight lines are easy to follow)
- Faster sampling at inference time

**The straightening procedure:**
1. Train a flow matching model (paths are slightly curved because the marginal vector field averages over all pairings)
2. Generate paired samples $(x_0, x_1)$ using the trained model
3. Retrain on these *new* pairs — the paths become straighter
4. Repeat

```
After 1 round:      After 2 rounds:     After 3 rounds:
   x₀      x₁          x₀    x₁           x₀   x₁
    ●──╮──╯●             ●──╮──●            ●────●
    ●──╮──╯●             ●───●              ●────●
    ●──╮──╯●             ●──╮●              ●────●
   (curved paths)       (straighter)       (straight!)
```

With straight paths, you can use very few steps (even 1 step for rough quality, 4-8 steps for high quality) — this is the core idea behind **Flux** and other modern image generators.

---

## Connection to Optimal Transport

Straight paths are related to **optimal transport** (OT). The OT coupling between $p_0$ and $p_1$ is the pairing that minimizes total travel distance. With OT pairing, paths are perfectly straight.

```
Random pairing:           OT pairing (minimal travel):
  x₀₁ ────────→ x₁₃        x₀₁ ──→ x₁₁  (nearby)
  x₀₂ ────────→ x₁₁        x₀₂ ──→ x₁₂
  x₀₃ ──────────────→ x₁₂  x₀₃ ──→ x₁₃
  (paths cross, longer)     (no crossing, shorter)
```

In high dimensions, computing exact OT is intractable, but approximations (minibatch OT) work well and produce near-straight paths from the start.

---

## Why Flow Matching Won

Before flow matching, diffusion models (DDPM) were the standard. Flow matching offers:

| | DDPM | Flow Matching |
|--|------|--------------|
| Training objective | Predict noise ε | Predict velocity (x₁ - x₀) |
| Path shape | Curved (cosine schedule) | Straight |
| Steps at inference | 100–1000 | 10–50 (often fewer) |
| Math complexity | Score functions, SDE theory | Simple interpolation |
| Quality | Excellent | Excellent (often better) |

Both **Stable Diffusion 3** and **Flux** (the best open-source image generators as of 2024) use flow matching (specifically rectified flow). The cleaner objective and faster sampling were the deciding factors.
