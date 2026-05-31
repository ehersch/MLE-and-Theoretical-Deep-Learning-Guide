# Image Classification Basics

The simplest CV problem: given an image, assign it a label from a fixed set. Getting this right required abandoning hand-crafted features and moving to a **data-driven** approach.

---

## The Data-Driven Paradigm

**Old approach (pre-2012):** write rules. Detect edges → group into shapes → match templates. Brittle, doesn't generalize.

**New approach:** collect labeled examples, learn a function that maps pixels to labels.

```
Training:                         Inference:
 (image₁, "cat")   ┐              new image
 (image₂, "dog")   ├─► learn f    ──────────► f(image) = "cat"
 (image₃, "cat")   ┘
```

The function $f$ is never hand-coded — it's fit to data.

---

## Dataset: CIFAR-10

The standard introductory benchmark.

```
50,000 training images
10,000 test images
10 classes: airplane, automobile, bird, cat, deer,
            dog, frog, horse, ship, truck
32×32 pixels, RGB

Each image: 32×32×3 = 3,072 numbers (integers 0–255)
```

Key insight: **the label is a global property of the image, not any single pixel.** A 32×32 image of a cat and the same image shifted by 1 pixel have identical labels but very different raw pixel vectors.

---

## K-Nearest Neighbors (KNN)

The simplest possible classifier: memorize training data. To classify a new image, find the $k$ nearest training images (by pixel distance) and take a majority vote.

```
Distance metrics:
  L1 (Manhattan): d(I₁, I₂) = Σᵢ |I₁ᵢ - I₂ᵢ|
  L2 (Euclidean): d(I₁, I₂) = √(Σᵢ (I₁ᵢ - I₂ᵢ)²)

k=1: label of the single nearest neighbor
k=5: majority vote among 5 nearest neighbors
```

```python
import numpy as np

class KNN:
    def fit(self, X_train, y_train):
        self.X, self.y = X_train, y_train
    
    def predict(self, X_test, k=1):
        # Vectorized L2 distances: (N_test, N_train)
        dists = np.sqrt(
            np.sum(X_test**2, axis=1, keepdims=True)
            - 2 * X_test @ self.X.T
            + np.sum(self.X**2, axis=1)
        )
        preds = []
        for i in range(len(X_test)):
            nn_idx = np.argsort(dists[i])[:k]
            preds.append(np.bincount(self.y[nn_idx]).argmax())
        return np.array(preds)
```

**Why KNN fails for images:**

1. **Speed:** classifying one test image requires comparing to all 50k training images. $O(N \cdot D)$ at test time — $N=50{,}000$, $D=3{,}072$.
2. **Curse of dimensionality:** in high dimensions, all points become equidistant. $k$ nearest neighbors stop being "near."
3. **Pixel distance is semantically meaningless:** an image shifted 1 pixel is distant in L2 but identical in content.

```
Two images that are "close" in L2 but semantically different:
  [cat on white background]  vs  [cat on white background, shifted 2px]
  L2 distance: ~100              These are clearly the same

Two images semantically similar but L2-distant:
  [brown cat]  vs  [orange cat]
  L2 distance: ~2000
```

KNN achieves ~35% on CIFAR-10. Random baseline: 10%. Best models: >99%.

---

## Linear Classifiers

The key abstraction: learn a score function that maps from pixels to class scores.

$$f(x; W, b) = Wx + b$$

- $x \in \mathbb{R}^D$: image flattened to a vector ($D = 32 \times 32 \times 3 = 3072$)
- $W \in \mathbb{R}^{C \times D}$: weight matrix ($C$ = number of classes = 10)
- $b \in \mathbb{R}^C$: bias vector
- Output: $C$ scores, one per class

```python
W = np.random.randn(10, 3072) * 0.01
b = np.zeros(10)

scores = W @ x + b   # (10,) — score for each class
pred   = np.argmax(scores)
```

### Three Viewpoints of Linear Classifiers

**1. Algebraic:** $f(x) = Wx + b$ — matrix multiply, each row of $W$ is a template.

**2. Visual (template matching):** each row $W_c$ reshaped to $32 \times 32 \times 3$ is a learned template for class $c$. Classification scores how much the input image looks like each template.

```
W["car"] reshaped:        W["horse"] reshaped:
┌──────────────┐          ┌──────────────┐
│  blurry car  │          │ blurry horse │
│  shape + avg │          │  shape + avg │
│  color       │          │  color       │
└──────────────┘          └──────────────┘
```

**3. Geometric:** each class defines a hyperplane in $\mathbb{R}^{3072}$. Classification = which side of each hyperplane does $x$ fall on?

**Fundamental limitation:** one template per class. A horse facing left and a horse facing right → the template averages them out and becomes a blurry two-headed horse. Can't handle intraclass variation.

---

## Loss Functions

We need to measure how bad our current $W$ is. A **loss function** $\mathcal{L}(W)$ should be low when scores are correct and high when wrong.

### Multiclass SVM Loss (Hinge Loss)

For image $i$ with correct class $y_i$ and scores $s = Wx_i$:

$$\mathcal{L}_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

where $\Delta = 1$ is a safety margin. For each wrong class $j$: if its score is less than the correct score by at least $\Delta$, no loss. Otherwise, we pay a linear penalty.

```
Scores for cat image:  cat=3.2  dog=5.1  ship=-1.7   (correct class: cat)
Δ = 1

Loss from dog:  max(0, 5.1 - 3.2 + 1) = max(0, 2.9) = 2.9
Loss from ship: max(0, -1.7 - 3.2 + 1) = max(0, -3.9) = 0
Total loss: 2.9 + 0 = 2.9   ← high! dog scored higher than cat
```

### Softmax (Cross-Entropy) Loss

Convert raw scores to probabilities via softmax, then take the negative log-probability of the correct class:

$$P(y_i \mid x_i) = \frac{e^{s_{y_i}}}{\sum_j e^{s_j}}$$

$$\mathcal{L}_i = -\log P(y_i \mid x_i) = -s_{y_i} + \log\sum_j e^{s_j}$$

```python
def softmax_loss(scores, y):
    # scores: (C,), y: int (correct class index)
    scores -= scores.max()           # numerical stability
    exp_s = np.exp(scores)
    probs = exp_s / exp_s.sum()
    return -np.log(probs[y])
```

**SVM vs Softmax:**
- SVM only cares about the correct score being higher by a margin — doesn't distinguish "very confident and right" from "barely right"
- Softmax always tries to push the correct class probability toward 1 — never saturates

In practice, both work similarly; softmax is more commonly used in deep learning.

---

## Regularization

Without regularization, the model can fit the training set perfectly (set $W$ to be enormous) but generalize poorly.

Add a regularization term to the loss:

$$\mathcal{L}(W) = \underbrace{\frac{1}{N}\sum_i \mathcal{L}_i}_{\text{data loss}} + \underbrace{\lambda R(W)}_{\text{regularization}}$$

- **L2:** $R(W) = \sum_{i,j} W_{i,j}^2$ — penalizes large weights, prefers diffuse solutions
- **L1:** $R(W) = \sum_{i,j} |W_{i,j}|$ — prefers sparse weights
- **Dropout, BatchNorm:** applied to activations, discussed in [cnn_architectures.md](cnn_architectures.md)

---

## The Full Training Pipeline

```
Data split:
  ┌─────────────────┬──────────┬─────────┐
  │   Training set  │  Val set │ Test set│
  │    (fit W)      │ (tune λ) │(report) │
  └─────────────────┴──────────┴─────────┘

For each hyperparameter setting:
  1. Train on training set
  2. Evaluate on val set
  3. Pick best hyperparameter
  4. Report ONCE on test set (never use test set for selection!)
```

**Why a validation set?** The test set can only be used once — it simulates deployment. Using it for hyperparameter search is "peeking" and inflates reported accuracy.

Linear classifiers achieve ~40% on CIFAR-10. Everything after this course is about building better score functions $f$.
