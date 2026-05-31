# Diffusion Models (Introduction)

Diffusion models are now the dominant paradigm for high-quality image generation, surpassing GANs in both quality and diversity. This is a conceptual introduction; the full mathematical treatment (SDEs, score matching, flow matching) is in the `Diffusion-and-Flow-Matching/` folder.

---

## The Core Intuition

**The forward process:** gradually add Gaussian noise to an image until it becomes pure noise.

**The reverse process:** learn to gradually remove that noise — starting from pure noise, denoise step by step back to a real image.

```
Forward process (q):  x₀ → x₁ → x₂ → ... → x_T ≈ N(0, I)
                    real   noisier  noisier     pure noise
                    image

Reverse process (p_θ): x_T → x_{T-1} → ... → x₁ → x₀
                      noise   cleaner   cleaner     generated image
                              ↑
                        neural network predicts how to denoise
```

The key insight: **learning to denoise at one noise level teaches useful structure about the data distribution at all levels.**

---

## Forward Process

At each step, add a small amount of Gaussian noise:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t;\; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t I)$$

where $\beta_t$ is a variance schedule (small, grows over $T$ steps).

**The magic property:** we can sample $x_t$ at any timestep $t$ directly from $x_0$ in closed form:

$$q(x_t | x_0) = \mathcal{N}(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t) I)$$

Or equivalently: $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$

where $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$.

```
t=0:   x_t = x_0             (pure image, ᾱ=1)
t=T/2: x_t ≈ blurry noise    (ᾱ ≈ 0.5)
t=T:   x_t ≈ N(0,I)          (pure noise, ᾱ≈0)
```

No neural network needed for the forward process — it's a fixed mathematical operation.

---

## The DDPM Training Objective

DDPM (Ho et al., 2020) simplifies the theoretical objective to a practical one: **train a neural network $\epsilon_\theta$ to predict the noise $\epsilon$ added at each step**.

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

Training algorithm:

```python
def ddpm_training_step(x0, model, T=1000):
    # Sample random timestep
    t = torch.randint(0, T, (x0.shape[0],))
    
    # Sample noise
    eps = torch.randn_like(x0)
    
    # Create noisy image at timestep t
    alpha_bar = get_alpha_bar(t)                    # precomputed schedule
    x_t = alpha_bar.sqrt() * x0 + (1-alpha_bar).sqrt() * eps
    
    # Predict the noise
    eps_pred = model(x_t, t)
    
    # Simple MSE loss
    return F.mse_loss(eps_pred, eps)
```

**Why predict noise instead of $x_0$ or $x_{t-1}$?** Empirically, noise prediction leads to more stable training and better sample quality (Ho et al., 2020 ablation).

---

## Reverse Process (Sampling)

To generate a new image:

```python
def ddpm_sample(model, shape, T=1000):
    x = torch.randn(shape)                     # start from pure noise
    
    for t in reversed(range(T)):
        t_tensor = torch.full((shape[0],), t)
        
        # Predict noise
        eps_pred = model(x, t_tensor)
        
        # Compute predicted x0
        alpha_bar_t = get_alpha_bar(t)
        x0_pred = (x - (1-alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
        
        # Denoise one step (compute x_{t-1})
        if t > 0:
            alpha_bar_prev = get_alpha_bar(t-1)
            mean = alpha_bar_prev.sqrt() * x0_pred + \
                   (1-alpha_bar_prev).sqrt() * eps_pred
            std = get_posterior_std(t)
            x = mean + std * torch.randn_like(x)
        else:
            x = x0_pred
    
    return x.clamp(-1, 1)
```

**The bottleneck:** 1000 forward passes through the neural network for one image — very slow. DDIM addresses this.

---

## DDIM: Faster Sampling

**Paper:** "Denoising Diffusion Implicit Models" (Song et al., 2020)

DDIM reformulates the reverse process as a **deterministic ODE** (instead of a stochastic SDE). This allows skipping timesteps:

```
DDPM: must take 1000 small steps (stochastic)
DDIM: can take 50-100 larger deterministic steps → 10-20× speedup

Quality vs speed tradeoff:
  1000 steps: highest quality
  100 steps:  near-indistinguishable quality  ← sweet spot
  20 steps:   good quality, visibly lower
  5 steps:    obvious artifacts
```

DDIM also enables **deterministic generation**: same noise → same image. Useful for editing (invert an image to noise, edit, denoise).

---

## The Neural Network Architecture

The neural network $\epsilon_\theta(x_t, t)$ that predicts noise typically uses a **U-Net** with time conditioning:

```
Input: x_t (noisy image, same resolution as output)
       t   (timestep embedding, sinusoidal → MLP)

U-Net:
  Encoder: [ResBlock + GroupNorm + SelfAttention] × L, downsampling
  Bottleneck: attention at lowest resolution
  Decoder: [ResBlock + GroupNorm + SelfAttention] × L, upsampling
            + skip connections from encoder
  Output: ε̂ (predicted noise, same shape as input)

Time conditioning: add time embedding to every ResBlock's hidden state
                   (elementwise addition after linear projection)
```

```
Time embedding:
  t → sinusoidal embedding (like transformer PE) → MLP → time vector τ
  
Each ResBlock:
  x → GroupNorm → Conv → SiLU → Conv → + x
       + τ (added to intermediate features)
```

---

## Classifier-Free Guidance (CFG)

Unconditional generation is interesting but we usually want **conditional** generation: "generate an image of a golden retriever."

**Classifier guidance** uses a separate classifier's gradients to steer generation — requires a noise-robust classifier, complex to train.

**Classifier-free guidance (Ho & Salimans, 2021):** train a single model for both conditional and unconditional generation by randomly dropping the condition during training (with probability 10–20%):

$$\hat{\epsilon}_\theta(x_t, c) = \underbrace{\epsilon_\theta(x_t, \emptyset)}_{\text{unconditional}} + w \cdot \underbrace{(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))}_{\text{condition direction}}$$

The **guidance scale** $w$ controls the strength:
- $w=0$: unconditional generation (diverse but sometimes off-condition)
- $w=7.5$: typical value (sharp, condition-faithful)
- $w=20$: very condition-adherent but less diverse, artifacts possible

```
High guidance (w=15):     Low guidance (w=1):
■■■■■■■■■■■■■■■           ░░░░░░░░░░░░░░░
"perfectly matches        "loosely resembles
the text prompt"           the prompt, diverse"
```

---

## Stable Diffusion: Latent Diffusion

**Paper:** "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)

**The problem:** running diffusion in pixel space at 512×512 is expensive. A 512×512×3 image has 786,432 values.

**Solution:** run diffusion in the **latent space** of a VAE.

```
Compression: 512×512×3 → VAE encoder → 64×64×4 latent (8× spatial downsampling)
                                              ↓
                               Diffusion in 64×64×4 latent space
                                              ↓
Expansion: 64×64×4 latent → VAE decoder → 512×512×3 image

Speed: 64×64 instead of 512×512 → 64× fewer diffusion operations
```

**Text conditioning via CLIP:** the text prompt is encoded by CLIP's text encoder. This conditioning is injected into the U-Net via **cross-attention** at multiple resolutions:

```
U-Net layer (at some spatial resolution H'×W'):
  Image features q ∈ ℝ^{H'W'×C}
  Text features  k, v ∈ ℝ^{77×768}   (77 CLIP text tokens)
  
  Cross-attention: softmax(qk^T/√C) · v
  → each spatial location attends to the most relevant text tokens
```

```
Stable Diffusion full pipeline:
  Text: "a photo of a golden retriever playing fetch"
    ↓ CLIP text encoder
  Text embedding (77 × 768)
    ↓
  Pure latent noise z_T ∈ ℝ^{64×64×4}
    ↓ U-Net (with text cross-attention) × 50 DDIM steps
  Denoised latent z_0
    ↓ VAE decoder
  Generated image ∈ ℝ^{512×512×3}
```

**FID comparisons (256×256 ImageNet):**

| Method | FID ↓ |
|--------|-------|
| BigGAN | 7.4 |
| StyleGAN2 | 3.8 |
| DDPM (Ho et al.) | 3.2 |
| ADM (improved DDPM) | 2.1 |
| LDM (Stable Diffusion) | 3.6 (at 4× compression) |
| DiT-XL/2 | 2.3 |

For the complete mathematical treatment of diffusion models, SDEs, flow matching, and architectures like DiT, see the `Diffusion-and-Flow-Matching/` section.
