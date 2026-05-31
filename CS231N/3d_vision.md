# 3D Vision

Images project 3D world onto 2D. 3D vision asks: how do we recover and reason about 3D geometry from images? The representations, from explicit meshes to implicit neural fields, define very different tradeoffs.

---

## 3D Shape Representations

There is no canonical "pixel" for 3D. Different representations suit different tasks.

### Voxels

Discretize 3D space into a grid, mark each cell occupied (1) or empty (0).

```
3D grid:  N × N × N binary array
N=32:   32,768 cells   ← tractable
N=256:  16M cells      ← expensive

Advantage:  Regular structure, easy to apply 3D convolutions
Drawback:   Resolution cubed — 10× finer resolution = 1000× more memory
            Mostly empty space (sparse structure)
```

### Point Clouds

Unordered set of 3D coordinates (and optionally normals, colors):

```
{(x₁,y₁,z₁), (x₂,y₂,z₂), ..., (xₙ,yₙ,zₙ)}

Example (LiDAR scan of a scene):
  100,000 to 1,000,000 points per scan

Advantage:  Compact, scales to large scenes
Drawback:   Unordered (no canonical arrangement) — need special architectures
            No connectivity — hard to define surfaces
```

### Meshes

Vertices + faces (triangular polygons connecting them):

```
Vertices: [(x₁,y₁,z₁), (x₂,y₂,z₂), ...]
Faces:    [(0,1,2), (1,2,3), ...]  ← triplets of vertex indices

Advantage:  Standard 3D graphics format, compact for smooth surfaces
            Renderable in real-time
Drawback:   Hard to learn (non-differentiable topology changes)
            Variable structure (different meshes have different connectivity)
```

### Implicit Functions (Neural Implicit Representations)

Represent shape as the zero-level set of a function $f: \mathbb{R}^3 \to \mathbb{R}$:

```
f(x,y,z) < 0:  interior
f(x,y,z) = 0:  surface (the shape boundary)
f(x,y,z) > 0:  exterior
```

The function is a neural network: continuous, memory-efficient, can represent arbitrary topology. Surface extracted via **marching cubes** algorithm.

**Signed Distance Function (SDF):** $f(p) = $ distance from $p$ to the nearest surface point (positive outside, negative inside).

---

## PointNet (2017)

**Paper:** "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" (Qi et al.)

**Challenge:** point clouds are **unordered sets**. A permutation of the same points is the same shape — the model must be permutation-invariant.

**PointNet's elegant solution:** apply the **same MLP** to each point independently (shared weights), then **max pool** across all points to aggregate.

```
Input: N × 3 (N points, each with x,y,z)

Per-point MLP (shared weights):
  N × 3 → N × 64 → N × 128 → N × 1024

Global max pool:
  N × 1024 → 1024  (global feature — permutation invariant!)

Classification head:
  1024 → 512 → 256 → k classes

Segmentation head (per-point):
  Concatenate global feature with per-point features
  N × (1024 + 64) → N × 512 → N × k
```

```python
class PointNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),   nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 1024))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256),  nn.ReLU(),
            nn.Linear(256, n_classes))
    
    def forward(self, pts):       # pts: (B, N, 3)
        feat = self.mlp(pts)      # (B, N, 1024)
        global_feat = feat.max(dim=1)[0]  # (B, 1024) — max pool
        return self.classifier(global_feat)
```

**Why max pool achieves permutation invariance:** $\max(f(p_1), f(p_2), ..., f(p_N))$ gives the same result regardless of the order of points.

**PointNet++:** hierarchical version. Groups nearby points into local regions, applies PointNet locally, subsamples, repeats. Captures local geometry that flat PointNet misses.

---

## NeRF: Neural Radiance Fields (2020)

**Paper:** "Representing Scenes as Neural Radiance Fields for View Synthesis" (Mildenhall et al.) — ECCV 2020 Best Paper

**The task:** given several photos of a scene from known viewpoints, synthesize novel views from any viewpoint.

```
Input:  ~100 photos of a scene, with known camera poses
Output: A model that can render the scene from any viewpoint

Application: capture a scene with a phone, render cinematic flythroughs
```

### Representation

NeRF represents the scene as a continuous 5D function (MLP):

$$F_\Theta(x, y, z, \theta, \phi) = (r, g, b, \sigma)$$

- Input: 3D position $(x,y,z)$ + viewing direction $(\theta, \phi)$
- Output: color $(r,g,b)$ + volume density $\sigma$

**Volume density $\sigma$:** how much the scene absorbs/emits light at this point. $\sigma \approx 0$: transparent (air). $\sigma \gg 0$: opaque (surface).

**Color** can depend on viewing direction (modeling specular/view-dependent effects).

### Volume Rendering

To render a pixel, shoot a ray through it and integrate the color along the ray:

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\, \sigma(\mathbf{r}(t))\, \mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt$$

where $T(t) = \exp\!\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) ds\right)$ is the **transmittance** (probability of not hitting anything before $t$).

In practice, discretize into $N=64$–$128$ sample points along the ray:

```python
def render_ray(model, ray_origin, ray_dir, near=2.0, far=6.0, N=64):
    # Sample points along ray
    t = torch.linspace(near, far, N)  # (N,)
    pts = ray_origin + t[:, None] * ray_dir  # (N, 3)
    dirs = ray_dir.expand(N, 3)
    
    # Query NeRF for each point
    rgb, sigma = model(pts, dirs)    # (N, 3), (N,)
    
    # Volume rendering
    delta = torch.cat([t[1:] - t[:-1], torch.tensor([1e10])])
    alpha = 1 - torch.exp(-sigma * delta)
    T = torch.cumprod(torch.cat([torch.ones(1), 1 - alpha[:-1]]), dim=0)
    
    C = (T * alpha).unsqueeze(-1) * rgb  # (N, 3)
    return C.sum(0)                       # (3,) — pixel color
```

**Training:** minimize MSE between rendered pixels and ground truth pixels across all training views. The NeRF MLP learns the scene geometry and appearance implicitly.

**Positional encoding:** raw $(x,y,z)$ coordinates are too low-frequency for the MLP to represent fine details. Map to higher-frequency features:

$$\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \sin(2^1\pi p), \cos(2^1\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$$

With $L=10$, this maps each coordinate to 20 features — the MLP can then learn high-frequency detail.

**Results and limitations:** stunning novel view synthesis quality. But:
- Slow to train: hours per scene
- Slow to render: minutes per image (must query MLP at every sample point along every ray)
- One model per scene (not generalizable)

Follow-ups (Instant-NGP, TensoRF) dramatically speed up training (seconds to minutes).

---

## 3D Gaussian Splatting (2023)

**Paper:** "3D Gaussian Splatting for Real-Time Novel View Synthesis" (Kerbl et al.) — SIGGRAPH 2023 Best Paper

**Motivation:** NeRF is slow. Gaussian splatting achieves better quality AND real-time rendering (>100fps).

**Representation:** instead of an implicit MLP, represent the scene as a collection of explicit 3D Gaussian "splats":

$$G_i(x) = \exp\!\left(-\frac{1}{2}(x - \mu_i)^\top \Sigma_i^{-1} (x - \mu_i)\right)$$

Each Gaussian $i$ has:
- Position $\mu_i \in \mathbb{R}^3$
- Covariance $\Sigma_i \in \mathbb{R}^{3\times3}$ (shape/orientation of the splat)
- Opacity $\alpha_i$
- Color (Spherical Harmonics coefficients for view-dependent color)

```
Scene = {(μᵢ, Σᵢ, αᵢ, colorᵢ)} for i = 1...N  (N ≈ 1-6 million Gaussians)

Rendering: project each 3D Gaussian onto the image plane → 2D ellipse
           sort by depth, alpha-composite front to back
           → pixel color

Differentiable rasterization: all steps differentiable → optimize with gradient descent
```

**Training:** start from a sparse point cloud (from COLMAP structure-from-motion on the input photos). Optimize Gaussian parameters to minimize photometric loss between rendered and real images.

**Adaptive densification:** split Gaussians that are too large; prune transparent ones.

**Comparison:**

| Method | Training time | Render time | Quality (PSNR) |
|--------|--------------|-------------|----------------|
| NeRF | 1-2 days | ~30 sec/frame | 31.0 dB |
| Instant-NGP | ~5 min | ~0.1 sec/frame | 32.0 dB |
| 3DGS | ~30 min | <0.01 sec (real-time!) | 33.3 dB |

3DGS is now the dominant method for novel view synthesis and is being extended to dynamic scenes, text-to-3D, and avatar generation.
