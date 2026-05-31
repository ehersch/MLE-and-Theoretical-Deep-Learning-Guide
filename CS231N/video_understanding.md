# Video Understanding

Images are 2D spatial signals. Video adds a **third dimension: time**. Understanding video requires modeling motion, temporal dependencies, and the relationship between appearance and dynamics.

---

## The Core Challenge

```
Image: H × W × 3               (single frame)
Video: T × H × W × 3           (T frames)

Example: 30fps, 10-second clip = 300 frames × 224 × 224 × 3 = ~40M values

Challenges:
  Motion: the same object looks very different across frames
  Long-term dependencies: action "diving" requires seeing the whole sequence
  Computational cost: T× more computation than images
  Temporal stride: actions happen at different speeds
```

---

## Benchmarks

**Kinetics-400/600/700:** large-scale action recognition.

```
Kinetics-400:  240k training clips, 400 action classes, 10-second clips
               Source: YouTube
               Classes: "playing guitar", "swimming", "cutting vegetables"

Kinetics-700:  650k training clips, 700 classes

Standard metric: Top-1 and Top-5 classification accuracy
Human performance on Kinetics-400: ~79%
Best models: >90%
```

**Something-Something v2:** 220k videos, 174 fine-grained action classes. Designed to require **temporal reasoning** (not just appearance):

```
"Pushing something so that it almost falls off"
vs
"Pushing something so that it falls off"

Same objects, same motion — temporal reasoning required to distinguish
→ Models with strong temporal modeling beat appearance-only models
```

**UCF-101:** 13,320 clips, 101 classes. Older but still used for transfer learning evaluation.

---

## Approach 1: 3D Convolutions (C3D)

**Key idea:** extend 2D spatial convolutions to 3D spatiotemporal convolutions.

```
2D Conv: kernel K × K × C_in      (spatial only)
3D Conv: kernel T × K × K × C_in  (spatiotemporal)
```

**C3D (Tran et al., 2015):** use 3×3×3 kernels throughout. A 3×3×3 conv on a $T \times H \times W$ volume mixes local spatial and temporal information simultaneously.

```
Input: T × 224 × 224 × 3    (e.g., T=16 frames)
  ↓ 3D Conv 3×3×3 (64)     → T × 112 × 112 × 64
  ↓ 3D Pool 1×2×2           → T × 56 × 56 × 64
  ↓ 3D Conv 3×3×3 (128)    → T × 56 × 56 × 128
  ↓ 3D Pool 2×2×2           → T/2 × 28 × 28 × 128
  ...
  ↓ FC layers → action class
```

**Cost:** 3D conv has $T$ times more parameters and computation than 2D conv. Computationally expensive.

---

## Approach 2: Two-Stream Networks (2014)

**Paper:** "Two-Stream Convolutional Networks for Action Recognition in Videos" (Simonyan & Zisserman)

**Insight:** separate appearance (what things look like) from motion (how they move). Process each with a separate CNN stream.

```
Video clip
   ↓                          ↓
RGB frames              Optical flow fields
(spatial stream)        (temporal stream)
   ↓                          ↓
CNN (VGG-16)            CNN (VGG-16)
   ↓                          ↓
Spatial score           Temporal score
         ↘             ↙
          Late fusion
              ↓
          Final prediction
```

**Optical flow:** a vector field representing apparent motion between consecutive frames. At each pixel, the flow vector $(u, v)$ indicates how much it moved from frame $t$ to $t+1$.

```
Frame t:                Frame t+1:
┌────────────┐          ┌────────────┐
│  ●  (ball) │   →      │    ●ʼ      │
└────────────┘          └────────────┘

Optical flow at ball location: (→, ↑) = the ball moved right and up
Flow field: dense motion map, one vector per pixel
```

**Why two streams?** Optical flow provides motion information even when appearance changes dramatically (camera shake, lighting changes). Spatial stream provides appearance context. Together they outperform either alone.

---

## Approach 3: SlowFast Networks (2019)

**Paper:** "SlowFast Networks for Video Recognition" (Feichtenbaum et al., Facebook AI)

**Biological motivation:** the human visual system has two pathways — one for sharp spatial detail (slow update), one for motion (fast update).

```
Slow pathway:    Processes 4 frames at full spatial resolution
                 → captures detailed appearance, semantics
                 → 80% of computation

Fast pathway:    Processes 32 frames at 1/4 spatial resolution
                 → captures temporal dynamics, motion
                 → 20% of computation

Lateral connections: fast → slow at multiple resolutions
Final: concatenate slow + fast features → classification
```

```
Input: 64-frame clip

Slow: sample every 16th frame → 4 frames × 224×224 → ResNet (α=8)
Fast: sample every 2nd frame  → 32 frames × 56×56  → small ResNet (β=1/8 channels)
                                       ↕  lateral connections
                              Fuse at multiple resolutions
                                       ↓
                              fc → action class
```

SlowFast achieves **79.8%** on Kinetics-400 with ResNet-50 backbone — much better than single-stream 3D CNNs.

---

## Approach 4: Video Transformers

**TimeSformer (2021):** apply transformer self-attention to video frames.

**The problem with naive application:** a 8-frame 224×224 video has $8 \times (224/16)^2 = 8 \times 196 = 1568$ patches. Self-attention is $O(1568^2) = 2.5M$ operations per head — expensive.

**Divided space-time attention:** factorize attention into temporal-only + spatial-only steps:

```
For each patch:
  Step 1: Temporal attention — attend to same spatial location across all frames
  Step 2: Spatial attention  — attend to all patches within the same frame

Cost: O(T·(H/P)²) instead of O((T·(H/P)²)²)
```

```
Video patches: [F1_P1, F1_P2, ..., F1_P196, F2_P1, ..., F8_P196]
                    ↑
TimeSformer temporal attention:
  For patch position (3,4):
    [F1_(3,4), F2_(3,4), ..., F8_(3,4)] ← same position across 8 frames
    Attention across these 8 tokens only
```

**ViViT (2021):** similar factorized attention, strong results with ViT backbone. ViViT-L achieves **83.0%** on Kinetics-400.

---

## Optical Flow Estimation

Traditional: Lucas-Kanade, Horn-Schunck (variational methods, slow).

**FlowNet (2015):** first end-to-end learned optical flow network. Encoder-decoder architecture with a correlation layer.

**RAFT (2020):** "Recurrent All-Pairs Field Transforms". Computes all-pairs correlation between feature maps of two frames, refines flow iteratively using a GRU.

```
Frame 1 → encoder → feature map f1  (H/8 × W/8 × C)
Frame 2 → encoder → feature map f2  (H/8 × W/8 × C)
                ↓
Correlation volume: C(f1, f2)[i,j,Δi,Δj] = f1[i,j] · f2[i+Δi, j+Δj]
  (for all displacements Δ in a search window)
                ↓
GRU × 12 iterations refining flow estimate
                ↓
Dense flow field (H × W × 2)
```

RAFT achieves state-of-the-art on optical flow benchmarks and is widely used for training video models.

---

## Architecture Comparison

```
Method              Kinetics-400   Params  Notes
─────────────────────────────────────────────────────
Two-Stream (VGG)    73.9%         138M×2  Requires optical flow
C3D (VGG)           82.3%         78M     Simple 3D conv
I3D (Inception)     74.2%         25M     "Inflated" 2D → 3D
SlowFast R50        79.8%         35M     Two temporal rates
SlowFast R101       79.8%         53M
TimeSformer-L       80.7%         121M    Transformer, no 3D conv
ViViT-L/16          83.0%         307M    Large ViT backbone
VideoMAE-H          86.6%         633M    SSL pretraining
```

**VideoMAE (2022):** masked autoencoder applied to video. Mask 90% of spatiotemporal patches and reconstruct — much more aggressive than image MAE (75%) because videos are highly redundant across frames. VideoMAE ViT-H achieves **86.6%** on Kinetics-400, the best published result.
