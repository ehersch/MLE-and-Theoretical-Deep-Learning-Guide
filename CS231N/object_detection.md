# Object Detection

Classification asks "what is in this image?" Detection asks "what is in this image, **and where**?" Each object gets a bounding box + class label.

---

## The Detection Problem

```
Input:  RGB image (arbitrary size)
Output: List of (class, confidence, x1, y1, x2, y2) for each detected object

Example:
  ┌─────────────────────────────────────┐
  │  ┌─────────┐          ┌──────────┐  │
  │  │  cat    │          │   dog    │  │
  │  │  0.97   │          │   0.89   │  │
  │  └─────────┘          └──────────┘  │
  └─────────────────────────────────────┘
  Output: [("cat", 0.97, x1,y1,x2,y2), ("dog", 0.89, x1,y1,x2,y2)]
```

Key challenges:
- **Variable output:** number of objects unknown at inference time
- **Localization:** need pixel-accurate box coordinates
- **Scale variation:** same object category appears at many scales
- **Overlapping objects:** multiple objects at the same location

---

## Benchmark: COCO

**MS COCO (Common Objects in Context):**

```
330,000 images
80 object categories (person, car, dog, pizza, ...)
1.5M object instances, with:
  - Bounding boxes (detection)
  - Instance segmentation masks
  - Keypoints (person pose)
  - Captions

Train: 118,287 images
Val:   5,000 images
Test:  40,775 images (labels withheld for competition)
```

**mAP (mean Average Precision):** the standard detection metric.

```
For each class:
  1. Rank detections by confidence (high → low)
  2. Mark each detection as TP (IoU > 0.5 with GT) or FP
  3. Compute Precision-Recall curve
  4. Average Precision (AP) = area under PR curve

mAP = mean AP across all 80 classes

COCO mAP = average over IoU thresholds 0.5:0.05:0.95
  (stricter than Pascal VOC which used IoU=0.5 only)
AP50 = AP at IoU threshold 0.5
AP75 = AP at IoU threshold 0.75 (stricter)
```

**Intersection over Union (IoU):**

```
        ┌────────────┐
        │  ┌─────────┼────┐
        │  │ Intersect│   │
        └──┼─────────┘   │
           └─────────────┘

IoU = Area(Intersection) / Area(Union)

IoU = 1.0: perfect overlap
IoU = 0.5: standard "good enough" threshold
IoU = 0.0: no overlap
```

---

## Non-Maximum Suppression (NMS)

Detectors often produce many overlapping boxes for the same object. NMS keeps only the best:

```python
def nms(boxes, scores, iou_threshold=0.5):
    """boxes: (N,4) in [x1,y1,x2,y2], scores: (N,)"""
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        # Compute IoU of this box with all remaining
        ious = compute_iou(boxes[i], boxes[order[1:]])
        # Keep boxes with IoU below threshold
        order = order[1:][ious < iou_threshold]
    return keep
```

---

## The Two-Stage Detector Arc

### R-CNN (2014)

**Paper:** "Rich feature hierarchies for accurate object detection" (Girshick et al.)

```
Image → Selective Search (2000 proposals) → For each proposal:
            ↓ warp to 227×227
            ↓ AlexNet forward pass
            ↓ SVM classifier
            ↓ Bounding box regressor
→ Detections
```

**Selective Search:** a classical (non-learned) algorithm that proposes ~2000 candidate regions based on color, texture, and size similarity. Fast but not learned.

**Problem:** 2000 CNN forward passes per image → ~47 seconds per image on GPU. Unusable in practice.

### Fast R-CNN (2015)

**Key insight:** run the CNN **once** on the whole image. For each proposal, extract features from the shared feature map using **RoI Pooling**.

```
Image → CNN backbone → Feature map (whole image at once)
                              ↓
Proposals (from Selective Search) → RoI Pool → fixed-size feature per region
                                                      ↓
                                           Shared FC → class + box refinement
```

**RoI Pooling:** given a region of interest (at any size/aspect ratio) in the feature map, divide it into a fixed $7×7$ grid and max-pool within each cell:

```
Proposal region in feature map (irregular size):
┌─────────────────────────┐
│                         │
│  [divide into 7×7 grid] │  → max pool each cell → 7×7×C fixed tensor
│                         │
└─────────────────────────┘
```

Result: **2.3 seconds per image**. Still slow because Selective Search (CPU) is the bottleneck.

### Faster R-CNN (2015)

**Paper:** "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (Ren et al.)

**Key insight:** why use Selective Search at all? The CNN features know more about the image than any classical algorithm. Learn the proposals too.

**Region Proposal Network (RPN):**

```
Feature map (from backbone, e.g., 13×13×512 for AlexNet on 224×224)
    ↓
3×3 conv sliding window
    ↓
For each of the 13×13 locations, predict k=9 anchors:
  - Objectness score (is there an object here?)
  - Box refinement (Δx, Δy, Δw, Δh relative to anchor)

Anchors: 3 scales × 3 aspect ratios = 9 boxes per location
Total: 13×13×9 = 1521 candidate boxes
NMS to top ~300 proposals
```

```
Full Faster R-CNN pipeline:
Image → Backbone → Feature map
                       ↓            ↓
                      RPN        RoI Pooling
                       ↓   proposals    ↓
                    top 300 → Box head → class + refined box
```

Faster R-CNN achieves **5fps** on GPU. This was the standard two-stage detector for years.

---

## Single-Stage Detectors

Two-stage: propose then classify. Single-stage: **directly predict boxes and classes from the feature map** in one pass.

### YOLO (You Only Look Once, 2015)

**Paper:** "You Only Look Once: Unified, Real-Time Object Detection" (Redmon et al.)

```
Divide image into S×S grid (S=7 for 448×448 input)

For each grid cell, predict:
  B bounding boxes (x, y, w, h, confidence) ← relative to cell
  C class probabilities (if object center falls in this cell)

Output tensor: S × S × (B×5 + C) = 7×7×(2×5+20) = 7×7×30

All in one forward pass → 45fps (vs Faster R-CNN's 5fps)
```

```
┌────┬────┬────┬────┬────┬────┬────┐
│    │    │    │    │    │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │  ▪ │    │    │    │    │    │  ← cell responsible for this dog
├────┼────┼────┼────┼────┼────┼────┤   (center of dog falls here)
│    │    │    │    │    │    │    │
└────┴────┴────┴────┴────┴────┴────┘
Each cell predicts 2 boxes + class probs
```

**Tradeoff:** much faster but misses small objects and clusters. Later versions (YOLOv3-v8) significantly improved accuracy while maintaining speed.

### SSD: Single Shot MultiBox Detector (2016)

Multi-scale prediction: predict boxes at multiple feature map scales. Shallow layers detect small objects (high resolution), deep layers detect large objects (low resolution, large receptive field).

---

## DETR: Detection Transformer (2020)

**Paper:** "End-to-End Object Detection with Transformers" (Carion et al., Facebook AI)

**Motivation:** R-CNN, YOLO, SSD all require hand-designed components (anchors, NMS, RPN). Can we eliminate all this?

```
Backbone (ResNet) → positional encoded feature map
    ↓
Transformer Encoder (self-attention over spatial features)
    ↓
Transformer Decoder (N=100 learnable "object queries")
    ↓ each query predicts one (class, box) or "no object"
N predictions (fixed set, N=100)
    ↓
Bipartite matching loss (Hungarian algorithm) → no NMS needed!
```

**Bipartite matching loss:** during training, find the optimal one-to-one assignment between the $N$ predictions and the ground truth objects (which are fewer than $N$). Unmatched predictions learn to predict "no object." This eliminates the need for NMS entirely.

```
Predictions:  [pred₁, pred₂, ..., pred₁₀₀]
GT objects:   [obj_A, obj_B]           (only 2 objects in image)

Hungarian algorithm finds: pred₇ → obj_A, pred₃₁ → obj_B
All others → "no object" class
```

**Results:** DETR matches Faster R-CNN on COCO AP but:
- No anchors, no NMS, no RPN — much simpler
- Handles global context better (transformers)
- Struggles with small objects (early versions)
- Slow to converge (500 epochs vs Faster R-CNN's 36)

Deformable DETR, DINO-det, and other follow-ups solve the convergence issue.

---

## Detection Performance Timeline (COCO AP)

```
Year  Method                        COCO AP
────────────────────────────────────────────
2015  Faster R-CNN (VGG-16)         21.9
2016  SSD (VGG-16)                  25.1
2017  Mask R-CNN (ResNet-101-FPN)   38.2
2018  YOLOv3                        33.0
2020  DETR (ResNet-101)             43.5
2021  Swin-L + HTC++                58.7
2022  DINO (Swin-L)                 63.3
2023  EVA + DINO                    66.0
```

The jump from ~25 to 63+ AP came from: FPN (multi-scale features), better backbones (ResNet→Swin→ViT), and transformer-based detectors.
