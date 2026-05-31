# Applications: Video, 3D, and Science

Diffusion and flow matching started with image generation but have spread to almost every domain where you need to model complex distributions. Here are the most impactful applications.

---

## Video Generation

Generating video requires generating **sequences of coherent frames** — every frame must look realistic, and consecutive frames must be temporally consistent (no flickering, realistic motion).

### The Core Challenge

```
Images: 512 × 512 × 3 = 786k dimensions
Video:  512 × 512 × 3 × 100 frames = 78M dimensions

Running diffusion in pixel video space: prohibitively expensive
Also: frames can't be generated independently (a running dog must
      have consistent appearance across all frames)
```

### Approach: Extend Latent Diffusion to 3D

The natural extension: compress video with a 3D VAE (spatial + temporal compression), then run diffusion on the compressed latent volume.

```
Video: T × H × W × 3
         ↓ 3D VAE encoder (compresses space AND time)
Latent:  T/4 × H/8 × W/8 × C   (e.g., 25 × 64 × 64 × 4)
         ↓ 3D diffusion model
Generated latent
         ↓ 3D VAE decoder
Generated video
```

### Extending Image DiT to Video

**The simplest approach:** take a pretrained image DiT (operating on 2D patches) and extend it to 3D (spatiotemporal patches).

A 16×16×16 spatiotemporal patch encodes a 1-second chunk of 16 spatial locations. These get flattened and processed by the transformer, which naturally handles relationships across time via self-attention.

```
Video latent: T × H × W × C
                ↓ patchify into spatiotemporal patches
(T/2) × (H/4) × (W/4) tokens   → flatten → sequence of tokens
                ↓ transformer (same as image DiT, just longer sequence)
predicted noise/velocity
```

### Lumiere (2024) — Google

Lumiere generates video by running diffusion at multiple temporal scales simultaneously (using a UNet-like temporal hierarchy). Key result: **temporal coherence** — no flickering.

```
Architecture:
  Slow stream: processes every 4th frame (global motion)
  Fast stream: processes every frame (local details)
  
  Similar to SlowFast (video understanding), but for generation
```

### Movie Gen (2024) — Meta

One of the largest video generation models. Key capabilities:
- 1080p video generation (very high resolution)
- Video editing: given an existing video + instruction, modify it
- Personalization: generate video of a specific person from a few photos
- Audio generation: jointly generate video + synchronized sound

Architecture: a very large DiT trained on video + text pairs, with flow matching.

### Genie (2024) — Google DeepMind

A different goal: **interactive world model**. Train on hours of video game footage, generate new game frames in response to actions.

```
Context:    previous 16 frames
Action:     user input (move left, jump, etc.)
Output:     next 16 frames consistent with the action

Applications:
  - Train RL agents in a learned simulator (no actual game needed)
  - Generate novel game levels
  - General "world simulator"
```

Genie represents a convergence of video generation and RL: a diffusion model trained on video becomes a simulator for agent training.

---

## 3D Generation

Generating 3D objects or scenes — used in game development, robotics simulation, and scientific visualization.

### DreamFusion: Text-to-3D via 2D Diffusion

**Paper:** Poole et al. (Google, 2022) — ICLR 2023 Best Paper

**The brilliant idea:** you don't need 3D training data. Use a pretrained 2D image diffusion model as a "3D geometry critic."

```
We want: a 3D model of a golden retriever

NeRF (neural radiance field): a function that can render the 3D object
                              from any viewpoint. Has parameters θ.

Process:
  1. Render the NeRF from a random camera viewpoint → image x
  2. Ask the diffusion model: "is this a good image of a golden retriever?"
  3. If not, compute Score Distillation Sampling (SDS) loss
  4. Gradient flows back into the NeRF parameters θ
  5. NeRF updates to look more like a golden retriever from that view
  6. Repeat for many random viewpoints
```

**Score Distillation Sampling (SDS):** the key technical contribution. The loss is:

```
Noise the NeRF render x to get x_t
Ask the diffusion model: "given x_t and prompt c, what noise was added?"
SDS gradient ≈ (predicted noise - actual noise added) × diffusion score

Intuition: push the render toward "what the diffusion model expects a golden
           retriever to look like" from every angle
```

**Results:** DreamFusion can generate 3D objects from text descriptions, consistent from all viewpoints. Quality is lower than image generation (the 2D diffusion model doesn't perfectly constrain 3D geometry) but impressive given no 3D training data.

### LRM: Large Reconstruction Model

**Paper:** Hong et al. (2024)

A different approach: train a model that takes **a single image** and outputs a full 3D model in a single forward pass.

```
Input:  one photo of a shoe (or any object)
Output: 3D NeRF of the shoe (can render from any angle)
Time:   ~5 seconds

Architecture:
  Image → DINO ViT → per-patch features
               ↓ cross-attention transformer
  Query tokens → 3D tri-plane NeRF representation
               ↓ volume rendering
  Rendered images (from multiple views) → training loss
```

LRM generalizes well because it's trained on millions of 3D assets (Objaverse, ShapeNet) with their corresponding 2D renders.

---

## Protein Design

Biology is where diffusion models may have their biggest impact. Proteins are long chains of amino acids that fold into 3D structures, and designing new proteins with desired functions is a central problem in drug discovery and biotechnology.

### RFDiffusion: Designing New Proteins

**Paper:** Watson et al. (Baker Lab, 2022)

**The task:** generate a protein backbone (3D arrangement of atoms) that will fold into a desired shape and potentially bind a target molecule.

```
Traditional protein design:
  1. Pick a target shape
  2. Search through possible sequences computationally
  3. Make many, test in the lab
  Time: months to years

RFDiffusion:
  Input:  target shape / function description
  Output: protein backbone (coordinates of each amino acid's Cα atom)
  Time:   seconds
```

**Architecture:** diffusion model operating on **protein backbone coordinates** $(x_1, y_1, z_1, ..., x_N, y_N, z_N)$ — essentially flow matching on points in 3D space.

A key challenge: protein structures are **rotation and translation invariant** (a protein folded in one orientation is the same as it rotated). RFDiffusion uses **equivariant** neural networks (SE(3)-equivariant) that respect this symmetry.

```
Standard neural network:  f(Rx) ≠ Rf(x)  (rotating input changes output unpredictably)
Equivariant network:      f(Rx) = Rf(x)  (rotating input just rotates the output)
```

Results from the Baker Lab: RFDiffusion-designed proteins bound therapeutic targets **with 10-100× better affinity** than proteins designed by previous methods. Some of these are now in clinical trials.

### AlphaFold 3

**Paper:** Abramson et al. (DeepMind, 2024)

AlphaFold 2 (2021) solved the protein structure prediction problem for single proteins. AlphaFold 3 extends this to **all biological molecules** — proteins, DNA, RNA, small molecules — and their interactions.

The key addition: AlphaFold 3 uses a **diffusion module** to generate the final atomic coordinates, rather than the geometric update module in AF2.

```
Input:  sequences and/or structure of all molecules in the system
Output: 3D coordinates of every atom

AlphaFold 3 diffusion module:
  Starts from random atomic positions
  Conditioned on sequence features (from a transformer)
  Runs ~200 diffusion steps
  Output: precise atomic coordinates

Applications:
  Drug discovery:     predict how a drug molecule binds a target protein
  Antibody design:    predict antibody-antigen binding
  DNA interactions:   predict how proteins bind DNA
```

AlphaFold 3 achieved state-of-the-art on protein-ligand docking by a significant margin — potentially accelerating drug discovery pipelines from years to months.

---

## Molecule Generation

Generating new small molecules — for drug discovery, materials science.

### Equivariant Diffusion for Molecules (EDM)

**Paper:** Hoogeboom et al. (2022)

Generate molecules as 3D point clouds: each atom has coordinates $(x,y,z)$ and a type (C, N, O, H, ...).

```
A molecule:
  Carbon (0.0, 0.0, 0.0)
  Carbon (0.0, 0.0, 1.2)
  Hydrogen (0.0, 1.0, -0.5)
  ...

Diffusion: add noise to both coordinates AND atom types
           (continuous noise on positions, discrete noise on atom types)
Generation: start from random positions/types, denoise
```

Uses equivariant networks (E(3)-equivariant) so the model doesn't care about orientation.

**Results:** EDM generates valid, drug-like molecules significantly better than previous approaches. The generated molecules often have desired properties (solubility, binding affinity) when conditioned on those properties.

---

## The Common Thread

Across all these applications, the recipe is the same:

```
1. Define the right DATA REPRESENTATION
   Images:    pixels or latents
   Video:     spatiotemporal latents
   Proteins:  3D backbone coordinates
   Molecules: 3D atom positions + types
   Text:      token sequences

2. Define the right CORRUPTION PROCESS
   Continuous data:  Gaussian noise (DDPM / flow matching)
   Discrete data:    masking or uniform noise
   3D structures:    must be rotation/translation equivariant

3. Train the RIGHT ARCHITECTURE
   Images:  U-Net or DiT
   3D data: equivariant networks (SE(3), E(3))
   Sequences: transformers

4. Use the right CONDITIONING
   Text: cross-attention to language model features
   Partial structure: condition on known parts, generate the rest
   Function: condition on desired properties
```

Diffusion and flow matching are general-purpose probability distribution modeling tools — wherever you can define a meaningful continuous representation, you can apply them.
