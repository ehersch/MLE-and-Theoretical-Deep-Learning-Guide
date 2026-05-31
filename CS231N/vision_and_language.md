# Vision and Language

The ultimate convergence: models that understand both images and text. This enables zero-shot recognition, image captioning, visual question answering, and multimodal generation.

---

## The Core Problem

```
Vision models:      understand images, but need labels
Language models:    understand text at internet scale
Vision-Language:    understand images THROUGH natural language supervision

Key insight: the internet has billions of (image, caption) pairs.
             This is free supervision for learning visual representations.
```

---

## CLIP: Contrastive Language-Image Pretraining

**Paper:** "Learning Transferable Visual Models from Natural Language Supervision" (Radford et al., OpenAI, 2021)

### Training

CLIP is trained on **400 million** (image, text) pairs scraped from the internet. The training objective is contrastive: match each image to its caption, push away non-matching pairs.

```
Batch of N (image, text) pairs:
  (img₁, text₁), (img₂, text₂), ..., (imgₙ, textₙ)

Image encoder (ViT or ResNet): img_i → e_i  (normalized embedding)
Text encoder  (Transformer):   text_j → t_j (normalized embedding)

Similarity matrix: S[i,j] = e_i · t_j   (N × N matrix)

Loss: cross-entropy along both rows and columns
  Row i:    S[i,i] should be highest in row i    (image → correct text)
  Column j: S[j,j] should be highest in col j    (text → correct image)

This is InfoNCE / NT-Xent loss over N² pairs:
  N positives: (eᵢ, tᵢ) — same pair
  N²-N negatives: all other (image, text) combinations
```

```
S = e · tᵀ  (N × N)

   text₁  text₂  text₃  text₄
img₁ [0.95  0.01   0.02   0.01]  ← high on diagonal (correct pair)
img₂ [0.02  0.93   0.03   0.01]
img₃ [0.01  0.02   0.96   0.01]
img₄ [0.01  0.01   0.02   0.94]
```

### Zero-Shot Classification

After CLIP training, classify any image into any categories — **without ever seeing those categories during training**:

```
Query: "What is in this image?" (dog, cat, or car?)

1. Encode image: e = image_encoder(img)

2. Encode text templates:
   t₁ = text_encoder("a photo of a dog")
   t₂ = text_encoder("a photo of a cat")
   t₃ = text_encoder("a photo of a car")

3. Predict: class = argmax_i (e · tᵢ)

No training needed! Just choose appropriate text prompts.
```

**CLIP zero-shot on ImageNet:** **76.2%** top-1 accuracy — matching ResNet-50 trained on ImageNet. Without seeing a single ImageNet training example.

### Why CLIP Is Transformative

```
Traditional:  Fixed categories, need labeled examples for each
CLIP:         Open-vocabulary — any concept describable in text

Examples:
  "Is this image safe for work?" → CLIP can answer
  "Find all photos of sunset over the ocean" → semantic search
  "Is there a cat with stripes in this image?" → compositional
  "Which photo looks more professional?" → subjective aesthetics
```

```python
import torch, clip
from PIL import Image

model, preprocess = clip.load("ViT-B/32")

image = preprocess(Image.open("dog.jpg")).unsqueeze(0)
text = clip.tokenize(["a photo of a dog", "a photo of a cat", "a photo of a car"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features  = model.encode_text(text)
    
    # Cosine similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features  /= text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(probs)  # e.g., [0.92, 0.06, 0.02]
```

---

## Image Captioning

Generate a natural language description of an image.

```
Input:  RGB image
Output: "A golden retriever playing fetch on the beach at sunset"
```

**Classic approach (CNN-LSTM):**

```
Image → CNN (ResNet) → feature vector
                            ↓
                      LSTM decoder:
                      Start: [feature vector]
                      Step 1: predict "A"
                      Step 2: predict "golden"
                      Step 3: predict "retriever"
                      ...
                      End: predict [EOS]
```

**Encoder-decoder with attention (2015):** instead of using a single global feature, the LSTM attends to different spatial regions of the CNN feature map when generating each word:

```
"golden" → attention weights on golden dog region
"beach"  → attention weights on sandy background
"sunset" → attention weights on horizon
```

---

## BLIP and BLIP-2

**BLIP (2022):** "Bootstrapping Language-Image Pre-training"

Key innovation: **generate captions for noisy web images and filter them** with a trained model. This "bootstrapping" produces cleaner training data from imperfect web captions.

**BLIP-2 (2023):** bridges frozen image encoders and frozen LLMs with a lightweight trainable module.

```
Architecture:
  Frozen image encoder (ViT-G)          ← not updated during training
         ↓ 257 visual tokens
  Q-Former (trainable, ~188M params)    ← learns to extract relevant visual info
         ↓ 32 query tokens               (bridges vision and language)
  Frozen LLM (OPT-6.7B or FlanT5)       ← not updated during training
         ↓
  Text output

Q-Former = transformer with:
  - Learnable query tokens that attend to image features
  - Self-attention between queries
  - Cross-attention to frozen image encoder
```

**Why frozen everything?** Training full models is expensive. BLIP-2 achieves strong VQA performance while training only 188M params (vs the 12B+ in the frozen models it uses).

---

## Flamingo (2022)

**Paper:** "Flamingo: a Visual Language Model for Few-Shot Learning" (Alayrac et al., DeepMind)

**Key capability:** few-shot multimodal learning. Given a few image-text examples in the context window, generalize to new examples.

```
Context window (interleaved images and text):

[image of cat] "This is a cat"
[image of dog] "This is a dog"
[image of ???] → model generates: "This is a rabbit"

Or:
[image + question] "What color is the car?" "Red"
[image + question] "What is the person doing?" "Running"
[new image + question] "What is in this bowl?" → "Pasta"
```

**Architecture:**

```
Pretrained LLM (Chinchilla 70B, frozen)
         ↕ cross-attention (trainable)
Pretrained vision encoder (NFNet, frozen)
  + perceiver resampler → fixed 64 visual tokens per image
```

The key is **interleaved** text and image processing in a single autoregressive model. Images are treated like special tokens in the text sequence.

---

## GPT-4V and Modern Multimodal LLMs

**GPT-4V (2023):** GPT-4 with vision capabilities. Can reason about images with chain-of-thought:

```
"Explain what's wrong with this code (screenshot)" → finds bugs
"What's the nutritional content of this meal?" → estimates from photo
"Solve this math problem (image of handwritten equation)" → solves it
"What would happen if I added these chemicals?" → safety reasoning
```

**LLaVA (Large Language and Vision Assistant, 2023):**

```
Image → CLIP ViT-L/14 (frozen) → 256 tokens
              ↓ linear projection (trained)
LLaMA-7B (fine-tuned on instruction-following with images)
              ↓
Text response
```

Surprisingly, this simple architecture (CLIP features + linear projection + LLaMA) achieves near GPT-4V performance when fine-tuned on good instruction data.

---

## Visual Question Answering (VQA)

```
Input:  Image + natural language question
Output: Answer (free-form or multiple choice)

Examples:
  [image of kitchen] "How many appliances are visible?" → "3"
  [image of chart] "What is the trend in sales?" → "Increasing from 2020 to 2022"
  [image of X-ray] "Is there evidence of pneumonia?" → "Yes, in the left lung"

Benchmark: VQA v2.0 — 200k images, 1.1M questions, human-verified answers
```

**VQA v2 results:**

| Model | VQA v2 accuracy |
|-------|----------------|
| LSTM + CNN (2016) | 58.2% |
| BERT + ResNet (2019) | 70.9% |
| CLIP + GPT (2021) | 76.9% |
| BLIP-2 (FlanT5-XXL) | 82.2% |
| GPT-4V | ~86% |

---

## The Vision-Language Ecosystem

```
Vision understanding:     CLIP features → linear probe → 76% ImageNet (no fine-tune!)
Generation:               CLIP conditioning → Stable Diffusion
Zero-shot retrieval:      embed images + queries, find nearest neighbors in CLIP space
Object detection:         CLIP + region proposals → open-vocabulary detection (GLIP, OWL-ViT)
Segmentation:             SAM (Segment Anything Model) + CLIP → open-vocabulary seg
Robotics:                 RT-2: CLIP + LLM → robot actions

The CLIP embedding space has become the "lingua franca" of vision-language:
  text ──► CLIP ──► embedding space ◄── CLIP ◄── image
                         ↕
                    can compare, retrieve, and generate
                    across both modalities
```

**Segment Anything Model (SAM, 2023):** Meta's foundation model for segmentation. Given any point, bounding box, or text prompt, segment the corresponding object. Trained on 1.1 billion masks.
