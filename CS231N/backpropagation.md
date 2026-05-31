# Backpropagation

We need $\nabla_W \mathcal{L}$ to do gradient descent. For a deep network with millions of parameters and a complex computation graph, computing this by hand for each parameter is impossible. **Backpropagation** is the efficient algorithm that does it automatically.

---

## Computational Graphs

Every computation can be represented as a directed acyclic graph (DAG) of elementary operations (nodes) with data flowing through edges.

```
Example: f(x, y, z) = (x + y) · z,  x=-2, y=5, z=-4

Forward pass:
  x=-2 ──┐
          ├──[+]──► q=3 ──┐
  y=5  ──┘                ├──[×]──► f=-12
                  z=-4 ───┘

Backward pass (chain rule in reverse):
  ∂f/∂f = 1
  ∂f/∂q = z = -4       ← multiply gate: gradient w.r.t. first input = second input
  ∂f/∂z = q = 3        ← multiply gate: gradient w.r.t. second input = first input
  ∂f/∂x = ∂f/∂q · ∂q/∂x = -4 · 1 = -4   ← add gate passes gradient through
  ∂f/∂y = ∂f/∂q · ∂q/∂y = -4 · 1 = -4
```

The key insight: **each node only needs to know its local gradient** (how its output changes with respect to its inputs), and the **upstream gradient** (how the loss changes with respect to its output). The chain rule does the rest.

---

## The Chain Rule in Matrix Form

For a composition $\mathcal{L} = f(g(x))$:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}$$

For vectors and matrices, the "gradient" of the loss w.r.t. a matrix $W$ has the **same shape as $W$**. The chain rule becomes:

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial z}^\top \cdot x^\top$$

where $z = Wx$ and $\partial \mathcal{L}/\partial z$ is the upstream gradient.

**Memory trick for matrix backprop:** the gradient of the loss w.r.t. any intermediate quantity always has the same shape as that quantity.

---

## Backprop Through Common Gates

### Add gate

$$z = x + y \implies \frac{\partial z}{\partial x} = 1, \quad \frac{\partial z}{\partial y} = 1$$

Gradient flows through unchanged (gradient distributor).

### Multiply gate

$$z = x \cdot y \implies \frac{\partial z}{\partial x} = y, \quad \frac{\partial z}{\partial y} = x$$

Inputs swap roles as gradients. Large input → large gradient on the other.

### Max gate

$$z = \max(x, y) \implies \frac{\partial z}{\partial x} = \mathbf{1}[x > y]$$

Routes gradient only to the "winner" (gradient switcher).

### Sigmoid

$$\sigma(x) = \frac{1}{1+e^{-x}}, \quad \frac{d\sigma}{dx} = \sigma(x)(1-\sigma(x))$$

```python
def sigmoid_forward(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(upstream_grad, x):
    s = sigmoid_forward(x)
    return upstream_grad * s * (1 - s)
```

### ReLU

$$\text{ReLU}(x) = \max(0, x), \quad \frac{d}{dx}\text{ReLU}(x) = \mathbf{1}[x > 0]$$

Dead ReLU problem: if $x < 0$ during forward pass, gradient = 0. If a neuron is always negative, it never updates ("dead neuron").

### Softmax + Cross-Entropy (combined)

$$\mathcal{L} = -\log\frac{e^{s_y}}{\sum_j e^{s_j}}, \quad \frac{\partial \mathcal{L}}{\partial s_j} = p_j - \mathbf{1}[j = y]$$

The gradient of softmax-CE loss w.r.t. logits is simply probabilities minus the one-hot label. Very clean to implement:

```python
def softmax_ce_backward(scores, y):
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    probs[y] -= 1          # subtract 1 from correct class
    return probs            # this is ∂L/∂scores
```

---

## Backprop Through a Linear Layer

Forward: $Z = XW + b$ where $X \in \mathbb{R}^{N \times D}$, $W \in \mathbb{R}^{D \times H}$, $b \in \mathbb{R}^H$

Given upstream gradient $\delta = \frac{\partial \mathcal{L}}{\partial Z} \in \mathbb{R}^{N \times H}$:

$$\frac{\partial \mathcal{L}}{\partial W} = X^\top \delta \in \mathbb{R}^{D \times H}$$
$$\frac{\partial \mathcal{L}}{\partial X} = \delta W^\top \in \mathbb{R}^{N \times D}$$
$$\frac{\partial \mathcal{L}}{\partial b} = \sum_i \delta_i \in \mathbb{R}^H \quad \text{(sum over batch)}$$

**Dimension check:** shapes always match the original parameter shapes. This is how you verify backprop equations.

---

## Full 2-Layer MLP from Scratch

```python
import numpy as np

class TwoLayerMLP:
    def __init__(self, D, H, C, lr=1e-3):
        self.W1 = np.random.randn(D, H) * np.sqrt(2.0/D)   # He init
        self.b1 = np.zeros(H)
        self.W2 = np.random.randn(H, C) * np.sqrt(2.0/H)
        self.b2 = np.zeros(C)
        self.lr = lr
    
    def forward(self, X, y=None):
        # Layer 1: linear + ReLU
        self.z1 = X @ self.W1 + self.b1      # (N, H)
        self.a1 = np.maximum(0, self.z1)      # (N, H) — ReLU
        # Layer 2: linear
        self.z2 = self.a1 @ self.W2 + self.b2 # (N, C)
        # Softmax
        exp_z = np.exp(self.z2 - self.z2.max(axis=1, keepdims=True))
        self.probs = exp_z / exp_z.sum(axis=1, keepdims=True)  # (N, C)
        
        if y is None:
            return self.probs.argmax(axis=1)
        
        # Cross-entropy loss
        N = X.shape[0]
        loss = -np.log(self.probs[np.arange(N), y]).mean()
        return loss, self.probs
    
    def backward(self, X, y):
        N = X.shape[0]
        
        # Gradient of loss w.r.t. z2 (softmax+CE combined)
        dz2 = self.probs.copy()
        dz2[np.arange(N), y] -= 1
        dz2 /= N                             # (N, C)
        
        # Gradients for W2, b2
        dW2 = self.a1.T @ dz2               # (H, C)
        db2 = dz2.sum(axis=0)               # (C,)
        
        # Backprop through layer 2 linear into a1
        da1 = dz2 @ self.W2.T              # (N, H)
        
        # Backprop through ReLU
        dz1 = da1 * (self.z1 > 0)          # (N, H) — zero out dead neurons
        
        # Gradients for W1, b1
        dW1 = X.T @ dz1                    # (D, H)
        db1 = dz1.sum(axis=0)              # (H,)
        
        # SGD update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
```

---

## Numerical Gradient Checking

Before trusting your backprop implementation, verify it numerically:

$$\frac{\partial \mathcal{L}}{\partial W_{ij}} \approx \frac{\mathcal{L}(W + h \cdot e_{ij}) - \mathcal{L}(W - h \cdot e_{ij})}{2h}$$

where $e_{ij}$ is a matrix with 1 at position $(i,j)$ and 0 elsewhere.

```python
def numerical_gradient(loss_fn, W, h=1e-5):
    grad = np.zeros_like(W)
    it = np.nditer(W, flags=['multi_index'])
    while not it.finished:
        ix = it.multi_index
        old = W[ix]
        W[ix] = old + h; L_plus  = loss_fn()
        W[ix] = old - h; L_minus = loss_fn()
        W[ix] = old
        grad[ix] = (L_plus - L_minus) / (2 * h)
        it.iternext()
    return grad

# Check: relative error should be < 1e-5
relative_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-10)
```

---

## Vanishing and Exploding Gradients

As networks get deeper, gradients can shrink or explode through repeated multiplication:

```
Depth 10, each layer's gradient = 0.1:  0.1^10 = 1e-10  ← vanished
Depth 10, each layer's gradient = 2.0:  2.0^10 = 1024   ← exploded
```

**Solutions:**
- **Weight initialization:** He init ($\sqrt{2/n_{\text{in}}}$) for ReLU keeps activation variance stable across layers
- **Batch normalization:** normalizes activations between layers (see [cnn_architectures.md](cnn_architectures.md))
- **Residual connections:** gradients can flow directly through skip connections
- **Gradient clipping:** for RNNs, clip $\|\nabla\|$ to a max norm

Modern deep networks (ResNet, ViT) are carefully designed so gradients flow well to all layers.
