# Score Matching and Diffusion Models

Diffusion models arrived before flow matching and are still widely used. The mathematical machinery behind them — score functions — is beautiful and connects to many other areas of probabilistic modeling.

---

## What Is a Score Function?

The **score** of a distribution $p(x)$ at a point $x$ is:

$$s(x) = \nabla_x \log p(x)$$

That's just the gradient of the log-density. It points in the direction where the probability density is increasing fastest.

```
Think of p(x) as a landscape:

        high density
         ╭────╮     ╭────╮
p(x)    /      \   /      \
────────           ─────────────

Score s(x) = slope of this landscape:
        ←←    ↑↑   →→  ←←   ↑↑  →→
(points uphill, toward peaks/modes)

At a mode (peak): s(x) = 0  (flat top)
Off a mode:       s(x) points toward the nearest peak
```

**Why is this useful for generation?** If you're at a random noise point, you can follow the score to move toward high-probability data regions — like following a hill uphill. This is **Langevin dynamics**:

$$x_{t+1} = x_t + \epsilon \cdot s(x_t) + \sqrt{2\epsilon}\, z, \quad z \sim \mathcal{N}(0,I)$$

Take small steps uphill (score direction) plus some noise (for exploration). Given enough steps with small enough $\epsilon$, this samples from $p(x)$.

**The problem:** we don't know $p(x)$ or its score $s(x)$. We only have samples.

---

## Denoising Score Matching

**Key insight (Vincent, 2011):** you don't need to estimate $s(x)$ directly. Instead, estimate the score of a **noisy version** of the data.

Given a noisy version $\tilde{x} = x + \sigma\epsilon$ where $\epsilon \sim \mathcal{N}(0,I)$, the score of the noisy distribution has a beautiful closed form:

$$\nabla_{\tilde{x}} \log p_\sigma(\tilde{x}) = -\frac{\tilde{x} - x}{\sigma^2} = -\frac{\epsilon}{\sigma}$$

This says: the score at a noisy point $\tilde{x}$ just points back toward the clean data point $x$. It's the direction to remove the noise.

**So train a neural network** $s_\theta(\tilde{x}, \sigma)$ to predict $-\epsilon/\sigma$ (the direction to denoise):

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{x, \epsilon, \sigma}\left[\left\| s_\theta(x + \sigma\epsilon, \sigma) + \frac{\epsilon}{\sigma} \right\|^2\right]$$

---

## DDPM: Denoising Diffusion Probabilistic Models

**Paper:** Ho et al. (2020) — the paper that triggered the modern diffusion era.

DDPM is essentially denoising score matching with a specific noise schedule and a practical training objective.

### The Forward Process

Add noise in $T=1000$ small steps. At each step, multiply by $\sqrt{1-\beta_t}$ and add $\sqrt{\beta_t}$ of noise:

$$x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon$$

The $\beta_t$ are small values that grow over time (the noise schedule). Because of repeated multiplication, by $t=T$ the image is almost pure Gaussian noise.

**The key trick:** you can jump directly to any timestep $t$ without running all $t$ steps one by one:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

where $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$ decreases from 1 to ≈0 as $t$ goes from 0 to $T$.

```
t=0:    x_t ≈ x_0          (ᾱ ≈ 1, almost no noise)
t=500:  x_t ≈ mix           (ᾱ ≈ 0.05)
t=1000: x_t ≈ N(0,I)        (ᾱ ≈ 0, pure noise)
```

### Training

Rather than predicting the score directly, DDPM trains the network to predict the **noise** $\epsilon$ that was added:

```python
def ddpm_training_step(model, x0):
    # 1. Sample a random timestep
    t = torch.randint(0, 1000, (x0.shape[0],))
    
    # 2. Sample random noise
    eps = torch.randn_like(x0)
    
    # 3. Create the noisy version (can do this in one shot)
    alpha_bar = get_alpha_bar(t)                          # precomputed
    x_t = alpha_bar.sqrt() * x0 + (1-alpha_bar).sqrt() * eps
    
    # 4. Train network to predict which noise was added
    eps_pred = model(x_t, t)
    
    return F.mse_loss(eps_pred, eps)
```

This is conceptually the same as denoising score matching — predicting the noise $\epsilon$ is equivalent to predicting the score (up to a scaling factor). Ho et al. found that predicting $\epsilon$ directly worked better empirically.

### Generation (Reverse Process)

Start from noise, iteratively remove it:

```
x_T ~ N(0,I)    ← start here
  ↓ predict ε, take one denoising step
x_{T-1}
  ↓
x_{T-2}
  ...
x_0             ← generated image

1000 steps total (one forward pass through the network per step)
```

Each denoising step:

```python
def ddpm_step(model, x_t, t):
    alpha_t     = get_alpha(t)
    alpha_bar_t = get_alpha_bar(t)
    beta_t      = 1 - alpha_t
    
    # Predict noise
    eps_pred = model(x_t, t)
    
    # Compute the denoised mean (predicted x_{t-1} center)
    mean = (x_t - beta_t / (1-alpha_bar_t).sqrt() * eps_pred) / alpha_t.sqrt()
    
    # Add a small amount of noise (unless t=0)
    if t > 0:
        noise = torch.randn_like(x_t)
        std   = beta_t.sqrt()
        return mean + std * noise
    return mean
```

---

## The Noise Schedule

The noise schedule $\{\beta_t\}$ controls how fast noise is added. Different schedules dramatically affect quality.

```
Linear schedule (original DDPM):
  β₁=1e-4, ..., β₁₀₀₀=0.02  (increases linearly)
  Problem: very noisy near t=T, wastes steps on "already destroyed" images

Cosine schedule (improved DDPM):
  ᾱ_t = cos²(π/2 · (t/T + s)/(1+s))
  Much smoother — images stay recognizable longer, more signal for learning

  ᾱ_t
   1.0 ╮
       │╲
       │  ╲
       │    ╲
   0.0 └───────► t
   (cosine — smooth drop)
```

The cosine schedule improved FID scores significantly by giving the model more useful training signal at high noise levels.

---

## NCSN: Noise-Conditional Score Networks

**Paper:** Song & Ermon (2019) — the paper that unified multiple noise levels into one model.

Rather than 1000 discrete noise levels like DDPM, NCSN trains a **single network** conditioned on the noise level $\sigma$:

$$s_\theta(x, \sigma) \approx \nabla_x \log p_\sigma(x)$$

Generation: start at high $\sigma$ (rough score), gradually reduce $\sigma$ while following the score (annealed Langevin dynamics).

This "score-based" view and the "DDPM" view are actually the same thing expressed differently — which Song et al. (2021) showed by unifying them under a **continuous-time SDE framework**.

---

## The Unification: SDEs Everywhere

**Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (2021)**

The key paper showing DDPM, NCSN, and flow matching are all special cases of one framework.

Any forward noising process can be written as an SDE:

$$dx = f(x,t)\,dt + g(t)\,dW$$

- **VP-SDE** (variance-preserving): corresponds to DDPM
- **VE-SDE** (variance-exploding): corresponds to NCSN
- **Flow ODE** (no noise): corresponds to flow matching

And any such SDE has a **reverse-time SDE** that runs backward (from noise to data) using the score:

$$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]\,dt + g(t)\,d\bar{W}$$

The only ingredient needed to reverse any of these processes: the score function $\nabla_x \log p_t(x)$.

```
The unifying picture:

Forward SDE:   data ──────── adds noise ──────────→ Gaussian
                              (fixed, no learning)
                                    ↕
             Learn score function ∇_x log p_t(x)
                                    ↕
Reverse SDE:   data ←─────── removes noise ──────── Gaussian
               (generated!)
```

---

## DDPM vs Flow Matching: Practical Comparison

```
                DDPM (noise prediction)    Flow Matching (velocity)
───────────────────────────────────────────────────────────────────
Forward path    Curved (noise schedule)   Straight (linear interp)
Training target Noise ε                   Velocity (x₁ - x₀)
Inference steps ~1000 (or ~50 with DDIM)  ~10-50 (even fewer w/ RF)
Mathematical    Score functions, SDEs     Simple interpolation
background
Modern usage    Stable Diffusion 1/2      SD3, Flux, many new models
```

Both work beautifully. Flow matching's straighter paths are the main practical advantage. For understanding the field, knowing both is essential.
