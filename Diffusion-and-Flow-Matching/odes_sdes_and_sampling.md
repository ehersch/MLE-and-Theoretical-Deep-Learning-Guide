# ODEs, SDEs, and Sampling

Before any neural network, we need to understand the continuous-time tools that diffusion and flow models are built on. These are simpler than they sound.

---

## The Core Problem: Sampling from Complex Distributions

We want to generate images that look like real photographs. A real photograph is a point in an extremely high-dimensional space (e.g., $512 \times 512 \times 3 = 786,432$ dimensions). The "distribution of real photos" $p_{\text{data}}(x)$ is a tiny, complicated manifold in that space.

We can't write down $p_{\text{data}}$ explicitly. But we can:
1. **Destroy** samples from $p_{\text{data}}$ by gradually adding noise → eventually reach a simple distribution (Gaussian) that we *can* sample from
2. **Learn to reverse** this destruction process

Both diffusion models and flow matching are ways of formalizing this destruction-then-reversal idea.

---

## Ordinary Differential Equations (ODEs)

An ODE describes how a point $x$ moves over time according to some velocity field $v$:

$$\frac{dx}{dt} = v(x, t)$$

Think of it like a particle floating in a river. The velocity field tells the particle which direction to move at each location and time.

```
Velocity field v(x, t):

    →  →  →  ↗  ↑
    →  →  ↗  ↑  ↑
    →  ↗  ↑  ↖  ←
    ↘  ↓  ↓  ←  ←
    ↓  ↓  ←  ←  ←

A particle starting anywhere gets carried by the field.
Different starting points trace different trajectories.
```

**Solving an ODE numerically (Euler's method):** take small steps in the direction of the velocity:

```python
def euler_solve(v_fn, x0, t_start=0.0, t_end=1.0, n_steps=100):
    x = x0.clone()
    dt = (t_end - t_start) / n_steps
    for i in range(n_steps):
        t = t_start + i * dt
        x = x + dt * v_fn(x, t)   # step in velocity direction
    return x
```

Simple, but not accurate for large steps. Better methods (RK4, adaptive step-size) are used in practice.

---

## How ODEs Generate Samples

Here's the key insight for generative modeling:

Suppose we design a velocity field $v(x, t)$ such that:
- At time $t=0$: $x$ is a sample from simple noise $p_{\text{noise}} = \mathcal{N}(0, I)$
- At time $t=1$: $x$ has been transported to a sample from $p_{\text{data}}$

```
t=0           t=0.25          t=0.5          t=0.75          t=1
●●●●●●●●   → ●●●●●●●●   → ●●●●●●●●   → ●●●●●●●●   → ●●●●●●●●
(Gaussian)                                              (face images)
   random blob        gradually organizing         structured data
```

If we have such a $v$, generation is:
1. Sample $x_0 \sim \mathcal{N}(0, I)$ (easy)
2. Run ODE from $t=0$ to $t=1$ using $v$ (just numerical integration)
3. $x_1$ is a sample from $p_{\text{data}}$

The question becomes: **how do we find the right $v$?** That's what flow matching and score matching answer.

---

## Stochastic Differential Equations (SDEs)

An SDE is an ODE with added randomness at every step:

$$dx = f(x, t)\, dt + g(t)\, dW$$

where $dW$ is "infinitesimal Gaussian noise" (a Wiener process — think of it as $\mathcal{N}(0, dt)$ noise added at each tiny step).

```
ODE trajectory:   smooth, deterministic
  ●────────────────────────────────────────→

SDE trajectory:   noisy, stochastic (different each time)
  ●─╮─╰─╮──╰──╮──╰──────────────────────→
    (same starting point, different random noise)
```

**Why use SDEs?** The added noise serves as a "thermostat" — it helps the process explore the distribution rather than getting stuck. During training, the forward process (adding noise to data) is an SDE. During generation, you can run either:
- The **reverse SDE** (noisy — more exploration, often higher quality)
- The corresponding **probability flow ODE** (deterministic — faster, reproducible)

Both produce samples from the same distribution.

---

## The Forward Process: Destroying Data

For diffusion models, the forward process gradually adds noise to a real image $x_0$ over time $t \in [0, T]$:

```
t=0         t=T/4       t=T/2       t=3T/4      t=T
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│ face │   │fuzzy │   │grain │   │ snow │   │noise │
│photo │   │ face │   │ face │   │      │   │      │
└──────┘   └──────┘   └──────┘   └──────┘   └──────┘

Forward SDE (adding noise):
  x₀ (data) ─────────────────────────────→ x_T ~ N(0,I)

Reverse SDE (removing noise):
  x_T (noise) ──────────────────────────→ x₀ (generated image)
```

The forward process is fixed (no learning) — it's just math. The neural network learns the reverse.

---

## The Reverse Process: From SDE to Neural Network

It turns out (Anderson, 1982) that **every forward SDE has a corresponding reverse SDE**. To run the reverse, you need one key ingredient: the **score function** $\nabla_x \log p_t(x)$.

The score is the gradient of the log-probability of the data at noise level $t$ — it points in the direction of "higher probability data":

```
Data distribution p_data (viewed as a landscape):
      ↑ density
      │    ╭────╮        ╭────╮
      │   /      \      /      \
      │──/────────\────/────────\──
                   x
                   
Score ∇_x log p(x): points UPHILL toward high-density regions

At a low-density point: score points toward the nearest mode
At a mode:             score is zero (local maximum)
```

If we know the score, we can denoise. If we can train a neural network to predict the score, we can generate samples. This is score matching, covered in [score_matching_and_diffusion.md](score_matching_and_diffusion.md).

---

## Summary: The Generative Modeling Problem as ODE/SDE

```
Goal: sample from p_data (complex, unknown)
      by learning to transform p_noise (Gaussian, easy to sample)

Tool:           What we learn:
────────────────────────────────────────────────
Flow matching   velocity field v(x,t)
Diffusion       score function ∇_x log p_t(x)
DDPM            noise ε added at each step

All three are equivalent ways of parameterizing the same transformation.
At the end of the day, we integrate an ODE/SDE from noise to data.
```

The next sections cover how each approach actually trains the neural network to learn this transformation.
