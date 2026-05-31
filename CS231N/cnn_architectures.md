# CNN Architectures

The story of CNN architectures from 2012–2020 is a story of discovering which design decisions actually matter. Each major architecture introduced one key idea that the field then widely adopted.

---

## Batch Normalization

Before going through specific architectures, BatchNorm is so fundamental that every architecture after 2015 uses it.

**The problem:** as training progresses, the distribution of each layer's inputs shifts as parameters of the previous layer change — **internal covariate shift**. The network constantly has to readjust to new input distributions.

**BatchNorm:** normalize each feature across the batch, then apply a learned scale $\gamma$ and shift $\beta$:

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

where $\mu_\mathcal{B}, \sigma_\mathcal{B}^2$ are batch statistics.

```
Without BN:                    With BN:
  Layer outputs drift           Layer outputs stay ~N(0,1)
  during training               → stable gradients
  → need tiny LR                → can use 10× higher LR
  → careful initialization      → less sensitive to init
  → slow convergence            → faster convergence
```

**Training vs inference:** during training, use batch statistics. During inference, use running averages $\bar{\mu}, \bar{\sigma}^2$ accumulated during training (so single-example inference works).

**Where to put it:** BN is typically placed after the linear/conv layer and before the activation:

```
Conv → BN → ReLU → ...    (standard)
Conv → ReLU → BN → ...    (also works, debated)
```

```python
class BatchNorm2d(nn.Module):
    def __init__(self, C, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta  = nn.Parameter(torch.zeros(C))
        self.register_buffer('running_mean', torch.zeros(C))
        self.register_buffer('running_var',  torch.ones(C))
        self.eps, self.momentum = eps, momentum
    
    def forward(self, x):          # x: (N, C, H, W)
        if self.training:
            mean = x.mean([0,2,3]) # mean over N, H, W for each channel
            var  = x.var([0,2,3], unbiased=False)
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
            self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*var
        else:
            mean, var = self.running_mean, self.running_var
        x_hat = (x - mean[None,:,None,None]) / (var[None,:,None,None] + self.eps).sqrt()
        return self.gamma[None,:,None,None] * x_hat + self.beta[None,:,None,None]
```

---

## Transfer Learning

Before building architectures from scratch, the most practical tool: **transfer learning**. A network pretrained on ImageNet has learned general visual features (edges, textures, object parts) that transfer to almost any vision task.

```
Pretrained on ImageNet (1.2M images, 1000 classes)
         │
         ▼
Your task (e.g., 500 images of dog breeds)

Strategy depends on your dataset size + similarity to ImageNet:
  ┌──────────────────────────────────────────────────────────┐
  │  Dataset size    │  Similar to ImageNet  │   Strategy    │
  ├──────────────────┼───────────────────────┼───────────────┤
  │  Small (<5k)     │  Yes                  │  Linear probe │
  │                  │                       │  (freeze all, │
  │                  │                       │   train FC)   │
  ├──────────────────┼───────────────────────┼───────────────┤
  │  Medium (~10k)   │  Yes/No               │  Fine-tune    │
  │                  │                       │  top layers   │
  ├──────────────────┼───────────────────────┼───────────────┤
  │  Large (>100k)   │  Yes/No               │  Full fine-   │
  │                  │                       │  tune or train│
  │                  │                       │  from scratch │
  └──────────────────┴───────────────────────┴───────────────┘
```

---

## Architecture Timeline

```
2012   AlexNet    ─── Proved deep CNNs work. Started the deep learning revolution.
2014   VGGNet     ─── Showed depth matters. Simple design, highly influential.
2014   GoogLeNet  ─── Width via Inception modules. 1×1 convolutions as a tool.
2015   ResNet     ─── Residual connections. Enabled training 100+ layer networks.
2017   DenseNet   ─── Each layer connected to all subsequent layers.
2018   SENet      ─── Channel attention ("squeeze-and-excitation").
2019   EfficientNet── Compound scaling of depth, width, resolution.
2021   ViT        ─── Transformers replace CNNs entirely. (see vision_transformers.md)
```

---

## AlexNet (2012)

See [seminal_papers_imagenet.md](seminal_papers_imagenet.md) for the full paper deep-dive.

```
Input (227×227×3)
    ↓ Conv 11×11, stride 4, 96 filters → 55×55×96
    ↓ MaxPool 3×3, stride 2            → 27×27×96
    ↓ Conv 5×5, pad 2, 256 filters     → 27×27×256
    ↓ MaxPool 3×3, stride 2            → 13×13×256
    ↓ Conv 3×3, pad 1, 384 filters     → 13×13×384
    ↓ Conv 3×3, pad 1, 384 filters     → 13×13×384
    ↓ Conv 3×3, pad 1, 256 filters     → 13×13×256
    ↓ MaxPool 3×3, stride 2            → 6×6×256
    ↓ FC 4096 + Dropout(0.5)
    ↓ FC 4096 + Dropout(0.5)
    ↓ FC 1000 + Softmax
```

**Key innovations:** ReLU (instead of sigmoid/tanh), Dropout, GPU training across 2 GPUs, data augmentation.

---

## VGGNet (2014)

See [seminal_papers_imagenet.md](seminal_papers_imagenet.md) for full deep-dive.

**The single design principle:** use only 3×3 filters. Stack many of them.

```
Two 3×3 convs  = same receptive field as one 5×5 conv
Three 3×3 convs = same receptive field as one 7×7 conv

But:
  Two 3×3:  2 × (3×3×C×C) = 18C² parameters
  One 5×5:  1 × (5×5×C×C) = 25C² parameters  ← 39% more params
  
  Plus: two layers = two non-linearities. One layer = one.
```

VGG-16 architecture:

```
[Conv 64] × 2 → Pool
[Conv 128] × 2 → Pool
[Conv 256] × 3 → Pool
[Conv 512] × 3 → Pool
[Conv 512] × 3 → Pool
FC 4096 → FC 4096 → FC 1000
138M parameters
```

---

## GoogLeNet / Inception (2014)

**Key question:** instead of going deeper (VGG), can we go wider?

**Inception module:** apply multiple filter sizes in parallel, concatenate results:

```
          Input feature map
         /       |       \       \
    1×1 conv  3×3 conv  5×5 conv  3×3 MaxPool
         \       |       /       /
          Concatenate along channel dim
                 ↓
          Output feature map
```

**Problem:** 5×5 convolutions on a 256-channel input are expensive: $5×5×256×256 = 1.6M$ ops per position.

**Solution: 1×1 bottleneck before expensive ops:**

```
Input (C=256)
    ↓  [1×1 conv → 32 channels]   ← reduce channels first
    ↓  [5×5 conv → 128 channels]  ← much cheaper: 5×5×32×128 = 204k
```

GoogLeNet used 22 layers but only 5M parameters (vs AlexNet's 60M, VGG's 138M).

**Auxiliary classifiers:** attached at intermediate layers to inject gradient signal (addresses vanishing gradients in deep networks before BatchNorm).

---

## ResNet (2015)

See [seminal_papers_imagenet.md](seminal_papers_imagenet.md) for full deep-dive.

**The degradation problem:** deeper networks should be at least as good as shallower ones (identity mapping is always a valid option). But in practice, 56-layer plain networks perform *worse* than 20-layer on CIFAR-10. This is not overfitting — training error is also higher.

**The hypothesis:** it's hard to learn identity mappings with multiple stacked non-linear layers.

**Residual connection:** instead of learning $H(x)$, learn the **residual** $F(x) = H(x) - x$, so $H(x) = F(x) + x$:

```
    x ─────────────────────────────────────────┐
    │                                           │
    ↓                                           │
[Conv → BN → ReLU → Conv → BN]                 │ (skip connection)
    │                                           │
    └─────────────[F(x)]──── + ─────────────────┘
                              │
                            ReLU
                              │
                           H(x) = F(x) + x
```

**Why does this work?**
- If the optimal function is close to identity, it's easier to push $F(x) \to 0$ than to learn $F(x) \to x$ from scratch
- Gradients can flow directly through the skip connection (no multiplication by weight matrices) → no vanishing gradients

**Bottleneck block (used in ResNet-50+):**

```
Input (256 channels)
    ↓ [1×1 conv → 64]     ← compress channels
    ↓ [3×3 conv → 64]     ← spatial convolution
    ↓ [1×1 conv → 256]    ← expand channels
    └────────────── + input (256 channels, via 1×1 conv if shapes differ)
```

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)          # skip connection
        return F.relu(out)
```

ResNet variants: ResNet-18, -34 (basic blocks), -50, -101, -152 (bottleneck blocks). ResNet-50 achieves 75.1% ImageNet top-1 with only 25M parameters.

---

## EfficientNet (2019)

**Observation:** model performance improves when you scale up width, depth, or resolution — but which to scale?

**Compound scaling:** scale all three together with a fixed ratio:

$$\text{depth}: d = \alpha^\phi, \quad \text{width}: w = \beta^\phi, \quad \text{resolution}: r = \gamma^\phi$$

where $\phi$ is the compound coefficient, and $\alpha, \beta, \gamma$ are found by grid search subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$.

EfficientNet-B7 achieves 84.3% top-1 on ImageNet with 66M parameters — comparable to models with 3–4× more parameters.

---

## Design Philosophy Summary

```
Architecture   Key idea              Parameters   ImageNet top-1
──────────────────────────────────────────────────────────────────
AlexNet        Deep + GPU            60M          63.3%
VGGNet         Depth via 3×3         138M         74.4%
GoogLeNet      Width via Inception    5M          74.8%
ResNet-50      Skip connections       25M         76.1%
ResNet-152     More depth             60M         77.8%
EfficientNet-B7 Compound scaling      66M         84.3%
ViT-H/14       Transformer (Sec 7)   632M         88.5%
```
