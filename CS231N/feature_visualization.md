# Feature Visualization and Interpretability

Neural networks are often treated as black boxes. Visualization tools let us peer inside and understand what features each layer and neuron has learned — and how robust (or fragile) those representations are.

---

## What Do Convolutional Filters Learn?

**Layer 1 filters** (visualized directly): interpretable, Gabor-like patterns — oriented edges and color blobs. Similar across all well-trained networks.

```
Example Conv1 filters (visualized as 11×11×3 images):
  [⟋ edge]  [⟍ edge]  [— horiz]  [| vert]
  [red+]    [blue+]   [green+]   [texture]
```

**Deeper layers:** filters respond to increasingly complex patterns — textures, object parts, whole objects. Not directly visualizable from weights alone — need activation-based methods.

---

## Saliency Maps

**Question:** which input pixels most influence the network's prediction for class $c$?

**Gradient-based saliency:** compute the gradient of the class score with respect to the input image:

$$S_c = \left|\frac{\partial f_c(x)}{\partial x}\right|$$

Large $|S_c(i,j)|$ means pixel $(i,j)$ is important for prediction.

```python
def compute_saliency(model, x, y):
    x = x.unsqueeze(0).requires_grad_(True)
    scores = model(x)
    score_c = scores[0, y]
    score_c.backward()
    saliency = x.grad.abs().max(dim=1)[0]  # max over channels
    return saliency.squeeze().detach()
```

```
Input image:          Saliency map:
┌────────────┐        ┌────────────┐
│            │        │  ░░░░░░░░  │  ← bright = important
│   [DOG]    │   →    │  ░░████░░  │
│            │        │  ░░████░░  │
└────────────┘        └────────────┘
```

**Limitation:** saliency maps are noisy and sensitive to small input perturbations. SmoothGrad (average gradients over noisy input copies) and Integrated Gradients (average gradients along a path from baseline to input) produce cleaner maps.

---

## Grad-CAM: Class Activation Mapping

**Paper:** Selvaraju et al. (2017)

Saliency is pixel-resolution but noisy. Grad-CAM produces **coarser but more semantic** heatmaps using the feature maps of the last convolutional layer.

**Algorithm:**

```
1. Forward pass → get logit for class c
2. Backprop to get gradients at last conv layer A^k (shape H×W×K)
3. Global average pool gradients: α_k^c = (1/HW) Σ_{i,j} ∂f_c/∂A^k_{ij}
4. Weighted combination: L_c = ReLU(Σ_k α_k^c · A^k)
5. Upsample L_c to input resolution → heatmap
```

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients, self.activations = None, None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o))
        target_layer.register_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0]))
    
    def __call__(self, x, class_idx):
        output = self.model(x)
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        weights = self.gradients.mean(dim=[2,3], keepdim=True)  # global avg pool
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        return F.relu(cam)  # only positive influence
```

```
Input:                    Grad-CAM (class: "cat"):
┌──────────────────┐     ┌──────────────────┐
│    cat   dog     │     │    ████          │  ← highlights the cat
│                  │  →  │    ████          │     not the dog
│                  │     │                  │
└──────────────────┘     └──────────────────┘
```

Grad-CAM is widely used for **model debugging** — if a classifier predicts "cat" but the heatmap lights up the background, the model may have learned spurious correlations.

---

## Activation Maximization (Feature Visualization)

**Question:** what image maximally activates a specific neuron or class score?

Start from random noise, optimize the input to maximize activation:

$$x^* = \arg\max_x f_c(x) - \lambda R(x)$$

where $R(x)$ is a regularization term (L2 norm, total variation) to keep the image natural-looking.

```python
def feature_visualization(model, class_idx, n_iter=500, lr=1.0):
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)
    
    for _ in range(n_iter):
        score = model(preprocess(x))[0, class_idx]
        loss = -score + 0.001 * x.norm()   # maximize score, minimize L2
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    return x.detach()
```

**What this reveals:**
- Shallow neurons: edges, gratings, color patches
- Mid-layer neurons: textures, object parts
- Class neurons: abstract "ideal" representations of each class

Early class neurons looked like abstract collages of the class features. With better regularization (total variation, image prior from a generative model), they look like recognizable objects.

---

## DeepDream

**Idea:** instead of maximizing a single neuron, **amplify** all the patterns a layer already detects in an image:

```python
def deepdream(model, image, target_layer, n_iter=30, lr=0.01, octaves=4):
    for octave in range(octaves):
        # Optionally scale image for multi-scale processing
        for _ in range(n_iter):
            image.requires_grad_(True)
            activations = get_layer_output(model, image, target_layer)
            loss = activations.norm()   # maximize activation magnitude
            loss.backward()
            with torch.no_grad():
                image += lr * image.grad / image.grad.abs().mean()
                image.clamp_(-1, 1)
    return image
```

The result: images with dreamlike, hallucinatory quality where textures from the training set are amplified recursively. DeepDream revealed that CNNs "see" dog faces everywhere (because ImageNet is dog-heavy).

---

## Neural Style Transfer

**Paper:** "A Neural Algorithm of Artistic Style" (Gatys et al., 2015)

Separate image **content** from **style** using CNN feature representations:

```
Content image (photo of Paris)  →  content representation (VGG-19 layer conv4_2)
Style image (Van Gogh painting) →  style representation (Gram matrices of multiple layers)

Optimization: find image x that matches BOTH representations
```

**Content loss:** match the feature map activations at a deep layer:

$$\mathcal{L}_{\text{content}} = \|F_l(x) - F_l(x_c)\|^2_F$$

**Style loss:** match the **Gram matrix** of feature maps at multiple layers:

$$G_l = F_l F_l^\top \in \mathbb{R}^{C_l \times C_l}$$

The Gram matrix captures **correlations between channels** — which feature types tend to co-occur — independent of spatial location. This captures texture and style without caring about where things are.

$$\mathcal{L}_{\text{style}} = \sum_l w_l \|G_l(x) - G_l(x_s)\|^2_F$$

```
Total loss: L = α·L_content + β·L_style
            (α/β ratio controls content vs style strength)

α/β = 1e-3:  strong style transfer (painting-like)
α/β = 1e-1:  weak style transfer (content-preserving)
```

```python
def gram_matrix(feat):
    B, C, H, W = feat.shape
    feat = feat.view(B, C, H*W)
    return feat @ feat.transpose(1, 2) / (C * H * W)

def style_loss(x_feats, style_feats):
    return sum(F.mse_loss(gram_matrix(x_f), gram_matrix(s_f))
               for x_f, s_f in zip(x_feats, style_feats))
```

---

## Adversarial Examples

**The alarming finding:** deep networks can be fooled by imperceptible perturbations of input images.

```
Original image: panda → 57.7% "panda"

+ tiny noise (ε = 0.007, invisible to humans):
→ 99.3% "gibbon"

The perturbation magnitude: max pixel change ≈ 0.007/255 ≈ nothing
```

### FGSM: Fast Gradient Sign Method

Goodfellow et al. (2014). Perturb the input in the direction that **increases the loss**:

$$x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(x, y))$$

One gradient step, very fast. Creates adversarial examples that are misclassified.

```python
def fgsm_attack(model, x, y, epsilon=0.01):
    x = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return (x + epsilon * x.grad.sign()).clamp(0, 1).detach()
```

### PGD: Projected Gradient Descent

Madry et al. (2018). Iterate FGSM multiple steps, projecting back into the $\epsilon$-ball after each step:

```python
def pgd_attack(model, x, y, epsilon=0.01, alpha=0.001, n_steps=40):
    x_adv = x.clone() + torch.randn_like(x) * epsilon  # random start
    for _ in range(n_steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        # Project back to epsilon-ball around original x
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon).clamp(0, 1).detach()
    return x_adv
```

### What Adversarial Examples Tell Us

CNNs are over-sensitive to high-frequency patterns that humans ignore. The network has learned features that are discriminative on the training distribution but are not the "semantic" features humans use.

**Adversarial training** (train on adversarial examples) improves robustness but costs ~2% clean accuracy. This accuracy-robustness tradeoff is an active research area.
