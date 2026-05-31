# Self-Supervised Learning

Labels are expensive. ImageNet took years and millions of dollars to annotate. **Self-supervised learning** (SSL) generates supervision from the data itself — no human labels needed. The goal: learn representations that transfer to downstream tasks.

---

## The Pretext Task Idea

Design a task where the label is derived automatically from the data:

```
Pretext task:          → Representation learned:
Predict rotation       → Orientation-invariant features
Solve jigsaw puzzle    → Spatial relationship understanding
Predict missing patch  → Context and semantics
Predict next frame     → Temporal dynamics
```

After pretraining on the pretext task, throw away the pretext head, fine-tune the backbone on your actual task. The backbone has already learned useful visual features.

---

## Contrastive Learning

The dominant SSL paradigm from 2020 onward. Core idea:

```
"augmented views of the same image should have similar representations;
 views from different images should have dissimilar representations"

Same image, different crops → PULL together in embedding space
Different images             → PUSH apart in embedding space
```

```
Image x                      Image y (different)
   ↓ augment ×2                  ↓ augment
  x₁       x₂                   y₁
   ↓ encode   ↓ encode           ↓ encode
  z₁         z₂                  z_y
  ← similar → ← ─ dissimilar ─ →
```

---

## SimCLR (2020)

**Paper:** "A Simple Framework for Contrastive Learning of Visual Representations" (Chen et al., Google Brain)

### Architecture

```
Image batch x
    ↓ Two random augmentations → x_i, x_j  (for each image in batch)
    ↓ Shared encoder f (ResNet-50)
    ↓ Projection head g (2-layer MLP) → z_i, z_j   ← contrastive loss here
```

**Augmentation strategy (critical for performance):**
```
Random crop + resize (most important!)
Random horizontal flip
Color jitter (brightness, contrast, saturation, hue)
Random grayscale
Gaussian blur
```

### NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)

For a batch of $N$ images → $2N$ augmented views. Each view $i$ has one positive (its pair $j$) and $2(N-1)$ negatives (all other views in the batch):

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

where $\text{sim}(u,v) = \frac{u \cdot v}{\|u\|\|v\|}$ (cosine similarity) and $\tau=0.1$ is the temperature.

```python
def nt_xent_loss(z1, z2, temperature=0.1):
    """z1, z2: (N, D) — paired augmented views"""
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)          # (2N, D)
    z = F.normalize(z, dim=1)
    
    # Similarity matrix (2N, 2N)
    sim = z @ z.T / temperature
    sim.fill_diagonal_(-1e9)  # remove self-similarity
    
    # Labels: for each view i, positive is i+N (or i-N)
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z.device)
    return F.cross_entropy(sim, labels)
```

**Key findings from SimCLR ablations:**
- Larger batch sizes → more negatives → better representations (batch 4096–8192 optimal)
- The projection head `g` matters: without it, accuracy drops ~10%
- Stronger augmentation → better representations (counterintuitively)
- Temperature $\tau$ critically impacts learning

**SimCLR-v2 results:** ResNet-50 pretrained with SimCLR → fine-tuned with 1% of ImageNet labels achieves **70.6%** top-1 (vs. 56.4% for supervised training with 1% labels).

---

## MoCo: Momentum Contrast (2020)

**Paper:** He et al. (Facebook AI)

**SimCLR's problem:** needs huge batches (4096–8192) to have enough negatives. Memory-intensive.

**MoCo's solution:** maintain a **memory bank** of negative keys from past batches, encoded by a **momentum encoder** (slowly moving average of the query encoder).

```
Query encoder (trained):  x_q → encoder_q → q
Key encoder (EMA, frozen): x_k → encoder_k → k

Memory queue: [k₋₁, k₋₂, ..., k₋Q]  (Q=65536 keys from past batches)
    ← dequeued oldest, enqueued newest each batch

Loss: InfoNCE contrastive loss
  q matched with k from same image (positive)
  q against all 65536 queue keys (negatives)
```

**Momentum update (key encoder):**

$$\theta_k \leftarrow m\theta_k + (1-m)\theta_q, \quad m = 0.999$$

The key encoder evolves very slowly → consistent representations across the queue (older keys are from a similar encoder → valid negatives).

**MoCo v2 + tricks from SimCLR:** MLP projection head + stronger augmentation → matches SimCLR performance with batch size 256 instead of 4096.

---

## BYOL: Bootstrap Your Own Latent (2020)

**Paper:** Grill et al. (DeepMind)

**The "no negatives" shocker.** BYOL achieves SOTA without any negative pairs at all.

```
Image x → augment → x₁    →  online encoder f_θ  → online projector g_θ  → q_θ
                                                           ↓ predictor h_θ
Image x → augment → x₂    →  target encoder f_ξ  → target projector g_ξ  → z_ξ

Loss: minimize ‖q_θ - stop_gradient(z_ξ)‖²  (MSE between normalized vectors)

Target network update: ξ ← τξ + (1-τ)θ  (EMA of online, τ=0.996)
```

**Why doesn't this collapse?** If both networks predict the same constant vector, loss = 0 — isn't that a valid solution?

Empirically, the EMA update + the predictor (an extra MLP only on the online side) create an **implicit contrastive signal** — but this is still actively debated. The asymmetry (predictor on one side, stop-gradient on the other) seems to be key.

**BYOL results:** ResNet-50 → **74.3%** ImageNet top-1 with linear probe — better than SimCLR at the time.

---

## DINO: Self-Distillation with No Labels (2021)

See [vision_transformers.md](vision_transformers.md) for the full treatment. Key summary:

```
Student (updated by gradient) ──────→ softmax prediction
    ↓                                      ↑
Teacher (EMA of student, no gradient) → softmax prediction (with centering)

Loss: cross-entropy(student_output, stop_gradient(teacher_output))
      over multiple crops (2 global + 8 local)

Teacher centering: subtract running mean of teacher outputs
                   prevents all-same-class collapse
```

DINO's most striking result: ViT features **without any labels** produce better segmentation of objects than supervised CNNs:

```
ViT-S/8 DINO self-attention → perfect object segmentation masks
→ indicates the model "understands" where objects are
```

---

## MAE: Masked Autoencoders (2021)

See [vision_transformers.md](vision_transformers.md) for details. SSL via reconstruction:

```
Mask 75% of patches → encoder processes visible 25% only
                    → lightweight decoder reconstructs masked patches
Loss: MSE in pixel space on masked patches only
```

**Performance:** MAE ViT-H → **87.8%** fine-tuned, **73.5%** with frozen features + linear probe.

MAE is much more efficient than contrastive learning: the encoder sees only 25% of patches per image → 4× more images per compute.

---

## SSL Evaluation Protocol

**Linear probing:** freeze backbone, train a linear classifier on top. Measures quality of frozen representation.

**Fine-tuning:** fine-tune the entire backbone with labels. Measures transfer performance.

**K-NN evaluation:** no training at all — nearest neighbor in embedding space. Measures raw representation quality.

```
Method          Linear probe    Fine-tune    k-NN
                (1% labels)    (100% labels) (100%)
─────────────────────────────────────────────────
SimCLR-v2       70.6%          79.8%         -
MoCo-v3         73.2%          83.2%         -
BYOL            74.3%          79.6%         -
DINO (ViT-S/8)  79.5%          82.8%        74.5%
MAE (ViT-H)     -              87.8%         -
```

SSL has largely closed the gap with supervised pretraining, and for large-scale pretraining (internet-scale data), SSL is now preferred.
