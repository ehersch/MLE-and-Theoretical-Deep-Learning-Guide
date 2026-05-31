# DDIM and Consistency Models

DDPM takes 1000 steps to generate an image. That's 1000 neural network forward passes — slow. This section covers the main strategies for dramatically speeding up sampling.

---

## The Speed Problem

```
DDPM:        1000 steps × (100ms/step) ≈ 100 seconds per image  ← unusable
DDIM:           50 steps × (100ms/step) ≈ 5 seconds             ← slow
DDIM:           10 steps × (100ms/step) ≈ 1 second              ← acceptable
Consistency:     1 step  × (100ms/step) ≈ 0.1 second            ← fast!
Distilled flow:  4 steps × (100ms/step) ≈ 0.4 second            ← sweet spot
```

Modern deployment targets are 1-8 steps. Getting there requires rethinking the sampling process.

---

## DDIM: Deterministic Sampling

**Paper:** "Denoising Diffusion Implicit Models" (Song et al., 2020)

DDPM's reverse process adds noise at each step (it's stochastic). DDIM removes that noise, turning the reverse process into a **deterministic ODE**.

The key insight: a DDPM-trained model can be reused with a different sampling process. You don't retrain — you just change how you sample.

**The DDIM update rule:**

```
DDPM (stochastic):
  x_{t-1} = μ(x_t, ε_pred) + σ_t · z,   z ~ N(0,I)
                                  ↑
                          added randomness

DDIM (deterministic, σ=0):
  x_{t-1} = √ᾱ_{t-1} · x̂₀ + √(1-ᾱ_{t-1}) · ε_pred
             ↑                   ↑
       "go toward          "keep some noise"
        predicted x₀"
```

**Why determinism helps:**
1. **Skip timesteps freely:** with a stochastic process, you can't skip steps without error. With DDIM's ODE, you can take much larger steps.
2. **Reproducible:** same noise $x_T$ → same output image every time (useful for editing)

```python
def ddim_step(model, x_t, t, t_prev, eta=0.0):
    """
    eta=0: fully deterministic DDIM
    eta=1: matches DDPM (full noise)
    """
    alpha_bar_t    = get_alpha_bar(t)
    alpha_bar_prev = get_alpha_bar(t_prev)
    
    # Predict noise at current timestep
    eps = model(x_t, t)
    
    # Predict x₀ from x_t and predicted noise
    x0_pred = (x_t - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
    
    # DDIM direction (pointing toward x_t from predicted x0)
    direction_xt = (1 - alpha_bar_prev - eta**2 * (1-alpha_bar_t)/(1-alpha_bar_prev)).sqrt() * eps
    
    # Optional stochastic noise
    noise = eta * (1-alpha_bar_prev).sqrt() / (1-alpha_bar_t).sqrt() * torch.randn_like(x_t)
    
    return alpha_bar_prev.sqrt() * x0_pred + direction_xt + noise
```

**DDIM with 50 steps matches DDPM with 1000 steps** in quality — a 20× speedup for free.

---

## ODE Solvers: Going Even Faster

DDIM is essentially Euler's method (the simplest ODE solver). Using better ODE solvers with adaptive step sizes can get high-quality images in **10–20 steps**.

**DPM-Solver (2022):** uses higher-order solvers tailored to diffusion model ODEs. Achieves DDIM-1000-step quality in just 10–20 steps.

```
Method         Steps for high quality   Notes
──────────────────────────────────────────────
DDPM           ~1000                    Stochastic
DDIM           ~50                      Deterministic
DPM-Solver++   ~10-20                   Higher-order ODE solver
DEIS           ~10                      Exponential integrator
Euler (FM)     ~25-50                   Flow matching, straighter paths
```

For flow matching models (SD3, Flux), the paths are straighter, so even simpler Euler integration works with fewer steps.

---

## Consistency Models

**Paper:** "Consistency Models" (Song et al., OpenAI, 2023)

DDIM and DPM-Solver still need multiple steps. **Can we generate high-quality images in a single step?**

Consistency models make this work by training a model with a very different objective.

### The Key Idea

Any point on the ODE trajectory from noise to data should map to the **same** final clean image $x_0$. A consistency model $f_\theta(x_t, t)$ should produce the same output regardless of which $x_t$ you start from:

```
ODE trajectory from noise to data:
  x_T ──────── x_{T/2} ──────── x_{T/4} ──────── x_0

Consistency property:
  f_θ(x_T, T) ≈ f_θ(x_{T/2}, T/2) ≈ f_θ(x_{T/4}, T/4) ≈ x_0

No matter where you "enter" the ODE trajectory, you always get x_0.
```

This is called the **self-consistency** condition.

### Training: Consistency Distillation

The easiest way to get a consistency model: **distill** from a pretrained DDPM.

```
Pretrained DDPM (teacher):
  Given x_t, can compute x_{t-Δt} via one DDPM step

Consistency distillation:
  Take two adjacent points on the trajectory: x_t and x_{t-Δt}
  Train f_θ so that: f_θ(x_t, t) ≈ f_θ(x_{t-Δt}, t-Δt)
  (both should map to the same x₀)
```

```python
def consistency_distillation_loss(teacher, student, x0, t_pairs):
    """
    t_pairs: list of (t, t-Δt) pairs
    teacher: pretrained DDPM
    student: consistency model being trained
    """
    total_loss = 0
    for t, t_prev in t_pairs:
        # Corrupt x0 to get x_t
        x_t = corrupt(x0, t)
        
        # One teacher step: x_t → x_{t-Δt}
        with torch.no_grad():
            x_t_prev = teacher.step(x_t, t, t_prev)
        
        # Student should give same prediction from both points
        pred_from_xt      = student(x_t,      t)
        pred_from_xt_prev = student(x_t_prev, t_prev)
        
        total_loss += F.mse_loss(pred_from_xt, pred_from_xt_prev.detach())
    return total_loss
```

### Sampling: One or a Few Steps

After training, generation is extremely fast:

```python
def consistency_sample_1step(model, shape):
    # Start from noise
    x_T = torch.randn(shape) * T
    # Single forward pass → clean image
    return model(x_T, T)

def consistency_sample_multistep(model, shape, timesteps=[T, T/2, T/4]):
    """Multi-step for higher quality"""
    x = torch.randn(shape) * T
    for t in timesteps:
        x0_pred = model(x, t)                    # predict clean image
        if t > timesteps[-1]:
            x = corrupt(x0_pred, next_t)         # add noise back
            # then re-denoise from next_t
    return x0_pred
```

**Multi-step consistency sampling:** jump to $x_0$ in one step, then add noise back at a lower level, then denoise again. Each "zigzag" improves quality:

```
Noise x_T
  ↓ one step → x̂₀  (rough estimate)
  ↓ add noise back to level T/2 → x_{T/2}
  ↓ one step → x̂₀  (better estimate)
  ↓ add noise back to level T/4 → x_{T/4}
  ↓ one step → final image
```

2-4 steps with this scheme matches DDIM's 50-step quality.

---

## Progressive Distillation

**Paper:** "Progressive Distillation for Fast Sampling" (Salimans & Ho, 2022)

An even simpler approach: train a student to match what the teacher does in **2 steps** in a single step, then repeat.

```
Round 1: Student₁ matches Teacher in 2 steps (500 steps → 1 step of Student₁)
Round 2: Student₂ matches Student₁ in 2 steps (500 → 250 → 125 → ... → 4)
Round 3: Student₃ matches Student₂ in 2 steps (4 → 2)
Round 4: Student₄ matches Student₃ in 2 steps (2 → 1)
```

After 4 rounds of distillation: 1-step generation matching 1000-step original quality. This is how **SDXL-Turbo** and similar models work.

---

## The Quality-Speed Tradeoff Landscape

```
Inference steps  →  Quality (FID, lower is better)

              1       4       8      20      50      100+
──────────────────────────────────────────────────────────
DDPM          —       —       —       —       —       3.2
DDIM          60      30      15      7       3.5     3.2
DPM-Solver    45      20      10      4       3.3     —
Consistency   25      8       5       4       —       —
Distilled FM  15      4       3.5     —       —       —
(Flux-schnell)
```

**Practical guidance:**
- **Best quality, time not a concern:** 50-step DDIM or DPM-Solver++ from a strong model (Flux-dev)
- **Fast generation, good quality:** 4-8 steps with distilled flow matching (Flux-schnell)  
- **Real-time/interactive:** 1-4 step consistency model
