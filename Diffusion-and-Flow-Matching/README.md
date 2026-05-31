# Diffusion and Flow Matching

A conceptual guide to modern generative models based on continuous dynamics — the math behind Stable Diffusion, DALL-E 3, Sora, Flux, and protein design tools like RFDiffusion. Based on MIT 6.S184.

---

## The Big Picture

All these models share the same underlying idea:

```
Define a process that DESTROYS structure (data → noise)
Learn to RUN IT BACKWARDS (noise → data)
```

The way you define "destroy" and "reverse" gives you different algorithms:

```
Diffusion models:   add Gaussian noise step by step → denoise step by step
Flow matching:      define a smooth path from noise to data → learn the vector field
Score matching:     learn the gradient of the data distribution → follow it uphill
```

They're all deeply connected — in fact, they turn out to be the same thing viewed differently.

---

## Contents

| File | MIT Lecture | Topics |
|------|-------------|--------|
| [odes_sdes_and_sampling.md](odes_sdes_and_sampling.md) | Lecture 1 | ODEs, SDEs, how continuous dynamics generate samples |
| [flow_matching.md](flow_matching.md) | Lecture 2 | Probability paths, vector fields, flow matching objective |
| [score_matching_and_diffusion.md](score_matching_and_diffusion.md) | Lecture 3-A | Score functions, DDPM, the SDE unification |
| [classifier_free_guidance.md](classifier_free_guidance.md) | Lecture 3-B | Conditional generation, CFG, guidance scale |
| [latent_spaces_and_architectures.md](latent_spaces_and_architectures.md) | Lecture 4 | VAEs, U-Net, DiT, Stable Diffusion, Flux |
| [discrete_diffusion.md](discrete_diffusion.md) | Lecture 5 | Diffusion over text tokens, masked diffusion, CTMCs |
| [ddim_and_consistency.md](ddim_and_consistency.md) | — | Fast sampling, consistency models, distillation |
| [applications.md](applications.md) | — | Video, 3D, proteins, molecules |

## Reading order

**Conceptual:** odes_sdes_and_sampling → flow_matching → score_matching_and_diffusion → classifier_free_guidance

**Practical:** latent_spaces_and_architectures → ddim_and_consistency → applications

**Language models:** discrete_diffusion
