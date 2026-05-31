# CS231N: Deep Learning for Computer Vision

Stanford's flagship computer vision course. Covers the full arc from pixels and linear classifiers through CNNs, transformers, detection, segmentation, self-supervised learning, generative models, and vision-language models.

---

## The Central Problem

```
Input:                          Output:
┌─────────────────┐             "a cat sitting on a mat"
│  3 × H × W     │    model     ──────────────────────────
│  integer array  │   ──────►   [0.92, 0.04, 0.04]  (cat/dog/bird)
│  (RGB pixels)   │             bounding boxes, masks, ...
└─────────────────┘
```

Raw pixels are just numbers. The field's central question: how do we learn representations that make visual understanding tractable?

---

## Contents

| File | Topics |
|------|--------|
| [image_classification_basics.md](image_classification_basics.md) | KNN, linear classifiers, SVM, softmax |
| [optimization.md](optimization.md) | SGD, momentum, Adam, LR schedules |
| [backpropagation.md](backpropagation.md) | Computational graphs, chain rule, NumPy MLP |
| [cnns.md](cnns.md) | Convolution, pooling, receptive fields |
| [cnn_architectures.md](cnn_architectures.md) | BatchNorm, AlexNet→ResNet, transfer learning |
| [seminal_papers_imagenet.md](seminal_papers_imagenet.md) | AlexNet, VGG, GoogLeNet, ResNet — deep dives |
| [vision_transformers.md](vision_transformers.md) | ViT, DeiT, Swin, DINO, MAE |
| [object_detection.md](object_detection.md) | R-CNN → Faster R-CNN → YOLO → DETR |
| [segmentation.md](segmentation.md) | FCN, U-Net, DeepLab, Mask R-CNN, panoptic |
| [self_supervised_learning.md](self_supervised_learning.md) | SimCLR, MoCo, BYOL, DINO, MAE |
| [generative_models_cv.md](generative_models_cv.md) | VAE, GAN, PixelCNN — FID/IS evaluation |
| [diffusion_models_intro.md](diffusion_models_intro.md) | DDPM intuition, DDIM, CFG, Stable Diffusion |
| [feature_visualization.md](feature_visualization.md) | Saliency, Grad-CAM, style transfer, adversarial examples |
| [video_understanding.md](video_understanding.md) | 3D CNNs, two-stream, SlowFast, TimeSformer |
| [3d_vision.md](3d_vision.md) | PointNet, NeRF, 3D Gaussian Splatting |
| [vision_and_language.md](vision_and_language.md) | CLIP, BLIP-2, Flamingo, GPT-4V |

## Reading order

**Core foundations:** image_classification_basics → optimization → backpropagation → cnns → cnn_architectures → seminal_papers_imagenet

**Modern architectures:** vision_transformers → object_detection → segmentation

**Learning paradigms:** self_supervised_learning → generative_models_cv → diffusion_models_intro

**Applications:** feature_visualization → video_understanding → 3d_vision → vision_and_language
