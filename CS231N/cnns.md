# Convolutional Neural Networks

The fully-connected MLP treats every input pixel as independent — the position of a feature doesn't matter to it. CNNs exploit the **spatial structure** of images using three key ideas: local connectivity, parameter sharing, and translation equivariance.

---

## Motivation: Why Not Just Use FC Layers?

For a 224×224 RGB image and a hidden layer of size 4096:

$$\text{FC layer parameters} = 224 \times 224 \times 3 \times 4096 = 616 \text{ million}$$

This is impractical, ignores spatial structure, and overfits massively. Images have strong local correlations — nearby pixels are related — and features like edges, textures, and shapes appear **everywhere** in an image, not just at fixed locations.

CNNs exploit this:
- **Local connectivity:** each neuron sees only a small spatial region (receptive field)
- **Parameter sharing:** the same filter is applied everywhere → detect the same feature anywhere
- **Translation equivariance:** if a cat moves right, the feature map moves right too

---

## The Convolution Operation

A **filter** (kernel) $W \in \mathbb{R}^{K \times K \times C_{\text{in}}}$ slides across the input volume and computes a dot product at each location.

```
Input (5×5×1):          Filter (3×3×1):       Output (3×3×1):
┌─────────────────┐     ┌───────────┐
│  1  2  3  0  1  │     │  0  1  0  │
│  0  1  2  3  1  │  ×  │  1  0  1  │  ──►   ┌─────────────┐
│  2  3  1  0  2  │     │  0  1  0  │        │ 7  9  8 ... │
│  1  0  2  1  3  │     └───────────┘        │ ...         │
│  3  2  1  0  1  │                          └─────────────┘
└─────────────────┘
    filter slides with stride 1, computing dot product at each position
```

Formally, for input $X$ and filter $W$, the output at position $(i, j)$:

$$Z_{i,j} = \sum_{k,l,c} W_{k,l,c} \cdot X_{i \cdot s + k,\; j \cdot s + l,\; c} + b$$

where $s$ is the stride.

**Output size formula:**

$$\text{Output size} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

where $W$ = input width, $K$ = kernel size, $P$ = padding, $S$ = stride.

```
Input 32×32, kernel 3×3, padding 1, stride 1:
  Output = (32 - 3 + 2×1)/1 + 1 = 32   ← "same" convolution, preserves size

Input 32×32, kernel 3×3, padding 0, stride 2:
  Output = (32 - 3 + 0)/2 + 1 = 15     ← spatial downsampling
```

---

## Multiple Filters and Channels

One filter produces one **feature map** (one "channel" of output). Use $C_{\text{out}}$ filters to produce $C_{\text{out}}$ feature maps:

```
Input:   H × W × C_in
Filters: K × K × C_in × C_out   (C_out different filters, each C_in-channel)
Output:  H' × W' × C_out
```

Parameters in one conv layer: $K \times K \times C_{\text{in}} \times C_{\text{out}} + C_{\text{out}}$ (bias)

```
Example: conv layer, 3×3 kernel, C_in=64, C_out=128
  Parameters = 3×3×64×128 + 128 = 73,856
  FC layer equivalent (64×64 spatial): 64×64×64 × 64×64×128 = ~17 billion
```

This is the power of parameter sharing — same filter applied across all spatial locations.

---

## 1×1 Convolutions

A special case: $K=1$. Acts as a pointwise linear transformation across channels. No spatial mixing — just mixes channels.

```
Input: H × W × C_in  ──[1×1 conv, C_out filters]──►  H × W × C_out
```

Used to: **reduce channels** (bottleneck in ResNet/Inception), project to a different channel dimension, add non-linearity without changing spatial size.

---

## Padding

Without padding, each convolution shrinks the spatial dimensions. With **zero-padding**:

```
"Valid" (no padding):  output shrinks by K-1 in each dimension
"Same" (padding=K//2): output = input size (for stride 1)

┌──────────────────────┐
│  0  0  0  0  0  0   │
│  0 ┌───────────┐ 0  │   ← zero-padded input
│  0 │ actual    │ 0  │
│  0 │ image     │ 0  │
│  0 └───────────┘ 0  │
│  0  0  0  0  0  0   │
└──────────────────────┘
```

Padding preserves spatial size through conv layers. Without it, a 10-layer network on a 32×32 image would reduce to 12×12 even with 3×3 kernels.

---

## Pooling

Pooling **downsamples** the spatial dimensions, reducing computation and providing some translation invariance.

**Max pooling (2×2, stride 2):**

```
Input 4×4:          After max pooling:
┌───────────────┐    ┌───────┐
│  1  3  2  4  │    │  3  4 │
│  5  6  1  2  │ ──►│  6  8 │
│  3  1  7  5  │    └───────┘
│  2  4  8  3  │
└───────────────┘

Each 2×2 region → max value
```

**Average pooling:** take the mean instead. Often used at the end of a network (global average pooling: pool the entire feature map to a single vector).

**Pooling has no parameters.** Stride 2 halves spatial dimensions; stride 1 doesn't change size.

---

## Receptive Field

The **receptive field** of a neuron is the region of the original input that influences its value.

```
Layer 1 (3×3 conv):  receptive field = 3×3
Layer 2 (3×3 conv):  receptive field = 5×5
Layer 3 (3×3 conv):  receptive field = 7×7
...
Layer k (3×3 conv):  receptive field = (2k+1) × (2k+1)

With 2×2 pooling (stride 2) after layer 2:
  Receptive field grows much faster
```

Deep networks have large receptive fields — neurons in the last conv layer "see" most of the image, encoding global context despite only using small local kernels at each layer.

**Dilated convolutions:** insert gaps in the filter to expand receptive field without pooling:

```
Standard 3×3:       Dilated 3×3 (dilation=2):   Dilated 3×3 (dilation=4):
■ ■ ■               ■ · ■ · ■                    ■ · · · ■ · · · ■
■ ■ ■               · · · · ·                    ...
■ ■ ■               ■ · ■ · ■                    Receptive field = 9×9
RF = 3×3            RF = 5×5                     RF = 13×13
```

Used in DeepLab for semantic segmentation — large receptive field with high resolution.

---

## The Full Conv Layer Forward Pass

```python
import numpy as np

def conv_forward_naive(x, w, b, stride=1, pad=1):
    """
    x: (N, C_in, H, W)
    w: (C_out, C_in, kH, kW)
    b: (C_out,)
    """
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
    out = np.zeros((N, C_out, H_out, W_out))
    
    for n in range(N):
        for c_out in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    h_start, w_start = i*stride, j*stride
                    patch = x_pad[n, :, h_start:h_start+kH, w_start:w_start+kW]
                    out[n, c_out, i, j] = (patch * w[c_out]).sum() + b[c_out]
    return out
```

In PyTorch this is a single call: `nn.Conv2d(C_in, C_out, kernel_size=K, stride=S, padding=P)`.

---

## Putting It Together: A CNN Architecture

```
Input: (N, 3, 32, 32)  ← batch of RGB images

[Conv 3→32, 3×3, pad=1]  →  (N, 32, 32, 32)
[ReLU]
[Conv 32→64, 3×3, pad=1] →  (N, 64, 32, 32)
[ReLU]
[MaxPool 2×2, stride=2]  →  (N, 64, 16, 16)

[Conv 64→128, 3×3, pad=1] → (N, 128, 16, 16)
[ReLU]
[MaxPool 2×2, stride=2]   → (N, 128, 8, 8)

[Global Average Pool]     → (N, 128)   ← spatial dims gone
[FC 128→10]               → (N, 10)    ← class scores
[Softmax]                 → (N, 10)    ← probabilities
```

Parameters: ~400k. FC-only equivalent on 32×32 images: ~50M. CNNs are ~100× more parameter-efficient.
