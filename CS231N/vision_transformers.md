# Vision Transformers

Transformers dominated NLP by 2020. The question was: do they work for vision? The answer was yes — but not without tricks — and by 2022 they surpassed CNNs on most benchmarks.

---

## From Language to Vision: The Patching Trick

Language transformers operate on sequences of tokens (words/subwords). Images don't have natural tokens. The key insight: **divide the image into patches, treat each patch as a token**.

```
Input image: 224×224×3
Patch size:  16×16

Number of patches: (224/16) × (224/16) = 14 × 14 = 196 patches
Each patch:  16×16×3 = 768 values → flatten → 768-dim vector
                                   → linear projection → D-dim embedding

Sequence fed to transformer: [CLS, patch₁, patch₂, ..., patch₁₉₆]
                              length: 197 tokens
```

The transformer has **no notion of spatial structure** — it processes a sequence. Position embeddings are added to each patch embedding to encode location.

---

## ViT: Vision Transformer

**Paper:** "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020)

### Architecture

```
Image (224×224×3)
    ↓ Split into 16×16 patches → (196, 768) flatten
    ↓ Linear projection E ∈ ℝ^{768×D}  → (196, D)
    ↓ Prepend [CLS] token       → (197, D)
    ↓ Add positional embeddings → (197, D)
    ↓
┌───────────────────────────────────────┐
│  Transformer Encoder × L              │
│  ┌─────────────────────────────────┐  │
│  │ LayerNorm                       │  │
│  │ Multi-Head Self-Attention        │  │
│  │ Residual + LayerNorm            │  │
│  │ MLP (GELU, 4×hidden)            │  │
│  │ Residual                        │  │
│  └─────────────────────────────────┘  │
└───────────────────────────────────────┘
    ↓ Take [CLS] token output
    ↓ MLP classification head
    → class prediction
```

### PyTorch Implementation

```python
import torch, torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        # Conv2d with kernel=stride=patch_size does the patching + projection
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)              # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1,2)  # (B, N, D)
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3,
                 n_classes=1000, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                activation='gelu', batch_first=True, norm_first=True)
            for _ in range(depth)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                   # (B, N+1, D)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x[:, 0])   # [CLS] token only
        return self.head(x)
```

### ViT variants

| Model | Depth | Heads | Dim | Params | ImageNet top-1 |
|-------|-------|-------|-----|--------|----------------|
| ViT-S/16 | 12 | 6 | 384 | 22M | 81.2% |
| ViT-B/16 | 12 | 12 | 768 | 86M | 83.1% |
| ViT-L/16 | 24 | 16 | 1024 | 307M | 85.2% |
| ViT-H/14 | 32 | 16 | 1280 | 632M | 88.5% |

### Why ViT needs more data

CNNs have built-in inductive biases: **translation equivariance** (same filter everywhere) and **locality** (features depend on local regions). These are correct priors for images — they constrain the function class.

ViT has no such biases: the attention mechanism can attend to any pair of patches. This is more expressive but requires more data to learn. ViT-B trained on ImageNet alone (1.2M images) underperforms ResNet-50. Trained on JFT-300M (300M images), it crushes ResNet.

---

## DeiT: Data-Efficient Image Transformers

**Paper:** Touvron et al. (Facebook AI, 2021)

**Question:** can we train ViT with only ImageNet (no JFT-300M)?

**Key idea: knowledge distillation from a CNN teacher**

Add a **distillation token** alongside the [CLS] token. This token is trained to predict the teacher (RegNet, a CNN) predictions instead of ground truth labels:

```
[CLS token] [distillation token] [patch₁] ... [patchN]
     ↓                ↓
 Cross-entropy    KL divergence with
 vs true label    teacher predictions
```

DeiT-B (86M params, trained on ImageNet-1k only) achieves **83.1%** — on par with ViT-B trained on JFT. CNNs as teachers dramatically improve ViT sample efficiency.

---

## Swin Transformer: Hierarchical ViT

**Paper:** Liu et al. (Microsoft, 2021) — ICCV 2021 Best Paper

**The ViT problem for dense tasks:** ViT processes all patches at full resolution throughout. For detection and segmentation, we need **multi-scale feature maps** (like CNN feature pyramids). Also, global self-attention is $O(N^2)$ — for a 512×512 image with 16×16 patches, $N=1024$ tokens → $10^6$ attention operations.

**Swin's solution: shifted windows**

```
Standard ViT: global attention (every patch attends to every patch)
              ─────────────────────────────────────────────────────

Swin: local window attention (each patch attends within a local window)
  ┌───┬───┬───┬───┐    ┌───┬───┬───┬───┐
  │ W1│ W2│ W3│ W4│    │ ░░│ ▓▓│ ▓▓│ ░░│  ← shifted window
  ├───┼───┼───┼───┤    ├───┼───┼───┼───┤      crosses boundaries
  │ W5│ W6│ W7│ W8│    │ ░░│ ░░│ ░░│ ░░│
  └───┴───┴───┴───┘    └───┴───┴───┴───┘
  Window attention       Shifted window attention
  (no cross-window)      (connects adjacent windows)
```

Two alternating attention types:
1. **W-MSA:** self-attention within each window (no cross-window communication)
2. **SW-MSA:** shifted windows — windows shift by half their size, enabling cross-window connections

**Hierarchical stages:** Swin merges patches as it goes deeper (patch merging = doubling channels, halving spatial resolution):

```
Input 224×224
Stage 1: 56×56, C=96     (4× downsampled from input)
Stage 2: 28×28, C=192    (8×)
Stage 3: 14×14, C=384    (16×)
Stage 4: 7×7,  C=768     (32×)
→ FPN or direct head for detection/segmentation
```

| Model | Params | ImageNet top-1 |
|-------|--------|----------------|
| Swin-T | 28M | 81.3% |
| Swin-S | 50M | 83.0% |
| Swin-B | 88M | 83.5% |
| Swin-L | 197M | 86.4% |

Swin-L achieves **58.7 box AP** on COCO detection (then SOTA), beating all CNN-based methods.

---

## DINO: Self-Supervised ViT

**Paper:** Caron et al. (Facebook AI, 2021)

DINO trains ViT without any labels using **self-distillation** — a student network matches the predictions of a teacher (which is an exponential moving average of the student's weights):

```
Image
  ↓ random crop (global view)   → student network → student_output
  ↓ random crop (global view)   → teacher network → teacher_output
                                   (EMA of student, no gradient)

Loss: cross-entropy(student_output, stop_grad(teacher_output))
      for multiple views (global + local crops)
```

**What DINO discovers without labels:**

```
DINO [CLS] token attention visualized:
  Input: photo of a dog
  Attention map: perfectly highlights the dog, ignores background

DINO features used for k-NN classification on ImageNet:
  ViT-S/8 (frozen):  74.5% top-1  ← no finetuning!
```

DINO's attention heads learn to segment objects without ever seeing a segmentation label — emergent behavior from scale and the self-distillation objective.

---

## MAE: Masked Autoencoders

**Paper:** He et al. (Facebook AI, 2021)

Inspired by BERT's masked language modeling: **mask 75% of image patches** and train a ViT to reconstruct them.

```
Input patches: ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
Masked (75%):  ■ □ □ ■ □ □ □ ■ □ □ ■ □ □ □ ■ □
                  ↓ encoder (only sees visible patches)
               [e₁, e₄, e₈, e₁₁, e₁₅]   ← sparse!
                  ↓ + mask tokens
               [e₁, M, M, e₄, M, M, M, e₈, ...]
                  ↓ lightweight decoder
               reconstructed pixel values for masked patches
```

**Why 75% masking?** Images are highly redundant — if you mask 25%, you can reconstruct patches from nearby context without understanding the image. 75% forces semantic understanding.

**MAE efficiency:** the encoder only processes 25% of patches → 4× faster training than ViT with all patches.

**Results:** MAE ViT-H (632M params) achieves **87.8%** on ImageNet after finetuning — comparable to supervised ViT-H but trained more efficiently. Also transfers better to downstream tasks.
