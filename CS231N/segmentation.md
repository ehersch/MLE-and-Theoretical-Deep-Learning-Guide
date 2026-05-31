# Segmentation

Detection gives bounding boxes. Segmentation gives **pixel-level** understanding. Three increasingly fine-grained tasks:

```
Semantic:       Instance:           Panoptic:
┌──────────┐    ┌──────────┐        ┌──────────┐
│ CAT CAT  │    │ cat₁ cat₂│        │ cat₁ cat₂│
│ CAT DOG  │    │ cat₁ dog₁│        │ cat₁ dog₁│
│ BG BG BG │    │ BG  BG BG│        │ sky  sky │
└──────────┘    └──────────┘        └──────────┘
Each pixel gets    Each instance      Every pixel labeled:
a class label      gets a unique      stuff (sky, grass) +
(no instances)     color/mask         things (cats, dogs)
```

---

## Benchmarks

**ADE20K:** 20,210 images, 150 semantic categories, dense annotations.

**Cityscapes:** 5,000 finely annotated urban driving scenes, 19 classes. Key for autonomous driving.

**Pascal VOC 2012:** 11,000 images, 20 classes + background.

**COCO panoptic:** 133 categories (80 things + 53 stuff).

**Metrics:**
- **mIoU** (mean Intersection over Union): average IoU across classes
- **Panoptic Quality (PQ):** $PQ = \underbrace{SQ}_{\text{seg quality}} \times \underbrace{RQ}_{\text{recognition quality}}$

---

## Semantic Segmentation

### FCN: Fully Convolutional Network (2015)

**Paper:** "Fully Convolutional Networks for Semantic Segmentation" (Long et al.)

**The key insight:** replace a classification network's FC layers with 1×1 convolutions. This makes the network **fully convolutional** — it can take any input size and produce a spatial output.

```
Classification CNN (e.g., VGG-16):
  Input: 224×224×3
  After 5 conv blocks: 7×7×512
  FC layers: 7×7×512 → 4096 → 4096 → 1000
  Output: 1 label

FCN (replace FC with 1×1 conv):
  Input: any H×W×3
  After 5 conv blocks: H/32 × W/32 × 512
  1×1 conv: H/32 × W/32 × C_classes
  Upsample (bilinear or deconv): H × W × C_classes
  Output: per-pixel class scores!
```

**Upsampling:**
- **Bilinear interpolation:** simple, no parameters
- **Transposed convolution (deconv):** learned upsampling (can introduce checkerboard artifacts)

**Skip connections in FCN:** combine coarse, semantic features (deep) with fine, spatial features (shallow):

```
Input → Conv blocks 1-3 → pool3 ──────────────────┐
                    ↓                              │
               Conv block 4 → pool4 ──────────────┐│
                    ↓                             ││
               Conv block 5 → pool5 → 1×1 conv   ││
                    ↓ ×32 upsample                ││
                    └── + pool4 × 2 upsample ─────┘│
                         └── + pool3 × 8 upsample ─┘
                                  ↓
                          FCN-8s (best)
```

FCN-8s achieves **62.7 mIoU** on Pascal VOC 2012.

---

### U-Net (2015)

**Paper:** "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al.)

Originally for medical image segmentation (with very few training images), U-Net became the dominant architecture for all dense prediction tasks.

```
Architecture (encoder-decoder with skip connections):

Encoder (contracting path):        Decoder (expanding path):
  Input: 572×572×1                   Output: 388×388×2

  [Conv×2, 64] → MaxPool            Upsample ← [Conv×2, 64] ←──┐
         ↓                                ↑                      │ (skip)
  [Conv×2, 128] → MaxPool          Upsample ← [Conv×2, 128] ←──┘
         ↓                                ↑
  [Conv×2, 256] → MaxPool          Upsample ← [Conv×2, 256] ←──┐
         ↓                                ↑                      │
  [Conv×2, 512] → MaxPool          Upsample ← [Conv×2, 512] ←──┘
         ↓                                ↑
       [Conv×2, 1024] (bottleneck) ────────
```

The "U" shape comes from the symmetric encoder-decoder. **Skip connections** concatenate encoder feature maps to decoder feature maps at matching resolution, preserving fine spatial detail that would otherwise be lost through pooling.

```python
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU())
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # Encoder
        self.enc1 = UNetBlock(3, 64);   self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetBlock(128,256); self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(256, 512)
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)  # 512 = 256 (up) + 256 (skip)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        self.head = nn.Conv2d(64, n_classes, 1)
    
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2)); b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
```

U-Net became the de facto standard for medical segmentation and remains competitive on general segmentation.

---

### DeepLab: Atrous Convolutions and CRF (2015–2018)

**The spatial resolution problem:** after 5 rounds of stride-2 pooling, the feature map is $\frac{1}{32}$ of the input. That's $7×7$ for a 224-pixel image — too coarse for segmentation.

**Solution: atrous (dilated) convolutions.** Remove the last 1–2 pooling layers and instead use dilated convolutions to maintain resolution while expanding receptive field.

```
Standard pooling:  conv → pool (stride 2) → conv → pool (stride 2)
  Feature map at 1/4, then 1/16 → coarse

Atrous:            conv → conv (dilation=2) → conv (dilation=4)
  Feature map stays at 1/4! Receptive field still grows.
```

**ASPP (Atrous Spatial Pyramid Pooling):** apply multiple dilations in parallel to capture multi-scale context:

```
Feature map
     ↓ 1×1 conv (rate=1)
     ↓ 3×3 conv (rate=6)   ← all in parallel
     ↓ 3×3 conv (rate=12)
     ↓ 3×3 conv (rate=18)
     ↓ Global avg pool
     └─ Concat → 1×1 conv → segmentation output
```

DeepLabv3+ achieves **89.0 mIoU** on Pascal VOC 2012 — a large jump over FCN's 62.7.

---

## Instance Segmentation

### Mask R-CNN (2017)

**Paper:** He et al. (Facebook AI) — ICCV 2017 Best Paper

**Extension of Faster R-CNN:** add a mask prediction branch in parallel with the classification and box regression branches.

```
Image → Backbone (ResNet-FPN) → Feature Pyramid
    ↓                               ↓
   RPN                         RoIAlign
    ↓ proposals                     ↓
    └──────────────────────────── Box head → class + box
                                   │
                                Mask head → per-class binary mask
                                   (28×28 output, per RoI)
```

**RoIAlign vs RoIPool:** RoI Pooling quantizes coordinates to integers → misalignment for small objects. RoIAlign uses bilinear interpolation at exact floating-point locations → better mask accuracy.

**Mask branch:** a small FCN applied to each RoI → $28 \times 28$ binary mask for each class independently (no competition between classes in the mask head).

```
RoI feature (14×14×256)
    ↓ Conv ×4 (256 channels each)
    ↓ Deconv ×2 (upsample to 28×28)
    ↓ Conv 1×1 → 80 channels (one binary mask per class)
Output: 28×28×80 — pick the predicted class's mask channel
```

Mask R-CNN achieves **37.1 AP (box) and 33.1 AP (mask)** on COCO — SOTA at the time. Easily extensible (add keypoint head → pose estimation).

---

## Panoptic Segmentation

Panoptic segmentation unifies semantic (stuff: sky, grass, road) and instance (things: people, cars) segmentation.

**Panoptic FPN (2019):** extend FPN with a semantic segmentation branch and merge with Mask R-CNN's instance output. Unify with a merging step that resolves conflicts (where stuff and things overlap).

**Mask2Former (2022):** unified transformer-based model for all segmentation types. Uses masked attention (each query only attends to its predicted region) → achieves SOTA on panoptic, instance, and semantic simultaneously.

---

## Segmentation Performance Timeline

```
Task: Semantic (Pascal VOC mIoU)
2015  FCN-8s (VGG)         62.7
2015  SegNet                60.1
2017  DeepLabv3 (ResNet)   85.7
2018  DeepLabv3+ (Xception) 89.0
2021  ViT-Adapter            90.5

Task: Instance (COCO mask AP)
2017  Mask R-CNN            33.1
2020  CopyPaste + ResNet    47.9
2022  Mask2Former (Swin-L)  57.8

Task: Panoptic (COCO PQ)
2019  Panoptic FPN          43.5
2022  Mask2Former           58.0
```
