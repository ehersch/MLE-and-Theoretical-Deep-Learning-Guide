# Latent Spaces and Neural Network Architectures

Knowing the training objective (predict noise or velocity) is only half the story. You also need:
1. **What space to run diffusion in** — pixel space is expensive
2. **What neural network architecture predicts the noise/velocity**

These choices determine the speed, quality, and resolution of the final model.

---

## Why Not Pixel Space?

Running diffusion directly on a 512×512 RGB image means the model operates on 786,432-dimensional vectors. Each denoising step is one forward pass through a large neural network on this full-resolution data.

Problems:
- Slow training (each batch is huge)
- Slow inference (each of 50 steps touches all 786k dimensions)
- Most variation in images is low-frequency (colors, shapes, semantics) — operating on every pixel is wasteful

**Solution: compress first, diffuse in the compressed space.**

---

## Latent Diffusion: The Core Idea

```
Original image (512×512×3 = 786k dims)
          ↓ VAE encoder (learned compression)
Latent    (64×64×4   = 16k dims)   ← 48× smaller!
          ↓ run diffusion here
Generated latent
          ↓ VAE decoder
Generated image (512×512×3)
```

The VAE (Variational Autoencoder) is trained separately to compress and decompress images. Once trained, its weights are frozen — diffusion training only learns to generate latents.

This is the core idea in **Stable Diffusion** (Rombach et al., 2022) and called **Latent Diffusion Models (LDM)**.

---

## The VAE Encoder/Decoder

The VAE learns a compact representation that:
- Captures all visually important information (reconstruction looks good)
- Is compressed enough to make diffusion feasible
- Has roughly Gaussian structure (good prior for diffusion)

```
Training the VAE:
  Image x → Encoder E → latent z ∈ ℝ^{64×64×4}
  Latent z → Decoder D → reconstructed x̂ ∈ ℝ^{512×512×3}
  
  Loss = ||x - x̂||² (reconstruction)
       + KL(q(z|x) || N(0,I)) (forces Gaussian structure)
       + perceptual loss (LPIPS) (looks good to a VGG)

After training:
  E and D are frozen
  Diffusion model operates entirely in z-space
```

**What the latent space looks like:** the 4 channels of a 64×64 latent aren't interpretable the way RGB channels are. They're learned features that encode shapes, textures, and colors compactly.

---

## Architecture 1: U-Net

The U-Net is the workhorse of diffusion models through Stable Diffusion 2. It takes a noisy latent (or image) and outputs the predicted noise or velocity.

```
Input: noisy latent z_t ∈ ℝ^{H×W×C}
       time embedding t
       text conditioning c (CLIP or T5 tokens)

U-Net architecture:
                 ┌─────────────────────────────────┐
                 │           Encoder               │
  64×64×C ──────►  ResBlock + Attention ──────────►  32×32×2C
                 │       ↕ skip connection          │
  32×32×2C ─────►  ResBlock + Attention ──────────►  16×16×4C
                 │       ↕ skip connection          │
  16×16×4C ─────►  ResBlock + Attention ──────────►  8×8×8C
                 └─────────────────────────────────┘
                              ↓ bottleneck
                         8×8×8C → attention → 8×8×8C
                              ↓
                 ┌─────────────────────────────────┐
                 │           Decoder               │
  8×8×8C ───────►  Upsample + ResBlock + Attention ─►  16×16×4C
                 │       ↕ skip connection          │
  16×16×4C ─────►  Upsample + ResBlock + Attention ─►  32×32×2C
                 │       ↕ skip connection          │
  32×32×2C ─────►  Upsample + ResBlock + Attention ─►  64×64×C
                 └─────────────────────────────────┘
                              ↓
               Output: predicted noise/velocity ε̂ or v̂
```

Key components:
- **ResBlocks:** convolutional layers with skip connections (for stable gradients)
- **Self-attention:** at lower resolutions (8×8, 16×16) — captures global structure
- **Cross-attention:** injects text conditioning. Each spatial location attends to the text tokens:

```
At resolution 8×8 (64 spatial tokens):
  Q = spatial features (from image)
  K, V = text features (from text encoder, 77 tokens)
  
  Output: each spatial location weighted sum of text tokens
  = "which text tokens are relevant to this image region?"
```

- **Time embedding:** timestep $t$ is encoded as a sinusoidal embedding, projected through an MLP, and **added** to each ResBlock's intermediate activations

---

## Architecture 2: Diffusion Transformer (DiT)

**Paper:** "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2022)

U-Nets are convolutional and don't scale as cleanly as transformers. DiT replaces the U-Net with a plain Vision Transformer (ViT).

```
Input: noisy latent z_t ∈ ℝ^{32×32×4}  (for SD3 / Flux)
       time t, text condition c

Step 1: Patchify the latent
  32×32×4 → 256 patches of size 2×2×4 = 256 × 16 tokens

Step 2: Add positional embeddings

Step 3: Standard transformer blocks × L
  ┌─────────────────────────────────────────────────────┐
  │ DiT block:                                          │
  │   LayerNorm (with adaLN conditioning)               │
  │   Self-attention                                    │
  │   LayerNorm (with adaLN conditioning)               │
  │   MLP                                               │
  └─────────────────────────────────────────────────────┘

Step 4: Unpatchify → output same shape as input
```

**adaLN (adaptive Layer Norm):** conditions each transformer block on the timestep $t$ and class/text label $c$:

```
Standard LayerNorm:   normalize, then scale by fixed γ, shift by β
adaLN:                normalize, then scale by learned γ(t,c), shift by β(t,c)
                       where γ,β are outputs of a small MLP from the condition

This lets the network "know" what timestep it's at and what to generate
without cross-attention (faster than cross-attention for class conditioning)
```

**Why DiT scales better:**

| Model | Architecture | FID ↓ | Parameters |
|-------|-------------|-------|------------|
| ADM (U-Net) | U-Net | 10.9 | 554M |
| DiT-XL/2 | Transformer | 2.3 | 675M |
| SD3 (MMDiT) | Transformer | — | 2B |

Transformers scale predictably with parameter count. U-Nets have a more complex dependency on architecture choices.

---

## MM-DiT: Multimodal DiT (Stable Diffusion 3, Flux)

**SD3 and Flux** extend DiT to handle text and image jointly. The key insight: process image patches and text tokens **together** in the same transformer, with full bidirectional attention.

```
Image tokens:  [v₁, v₂, ..., v₂₅₆]    (from patchified latent)
Text tokens:   [t₁, t₂, ..., t₇₇]     (from T5 text encoder)

Concatenate:   [v₁...v₂₅₆, t₁...t₇₇]   → 333 tokens total

Self-attention:  every image token can attend to every text token and vice versa
                 → much richer image-text interaction than cross-attention

Separate weight matrices for image vs text tokens:
  Image path: W_Q^img, W_K^img, W_V^img
  Text path:  W_Q^txt, W_K^txt, W_V^txt
  But they share the same attention operation
  → text and image "talk to each other" directly
```

This is why SD3 and Flux follow text prompts so much better than SD1/SD2 — the text conditioning is far more integrated.

---

## Complete Stable Diffusion Pipeline

```
TEXT: "a photo of a golden retriever on the beach"
  ↓ CLIP text encoder → 77 × 768 text embeddings
  ↓ (optional) T5 text encoder → 77 × 4096 text embeddings (SD3/Flux)

NOISE: sample z_T ~ N(0, I) ∈ ℝ^{64×64×4}

DENOISING LOOP (50 steps):
  For t = T, T-1, ..., 1:
    ε_pred = U-Net(z_t, t, text_embeddings)    [or DiT]
    z_{t-1} = denoise_step(z_t, ε_pred, t)     [DDIM or Euler step]

DECODE:
  z_0 ∈ ℝ^{64×64×4} → VAE decoder → image ∈ ℝ^{512×512×3}
```

---

## Flux: The Current State of the Art

**Flux** (Black Forest Labs, 2024) uses:
- **Flow matching** (rectified flow) instead of DDPM
- **MM-DiT** transformer architecture (like SD3)
- **Rotary positional embeddings** (RoPE) for better position awareness
- Parallel streams for text and image (then joint processing)
- Two variants: Flux.1-dev (high quality, 50 steps) and Flux.1-schnell (distilled, 1-4 steps)

Flux represents the convergence of ideas: flow matching + DiT + joint text-image attention + distillation for speed.
