# Optimization

We have a loss function $\mathcal{L}(W)$. We want the $W$ that minimizes it. For neural networks, there is no closed-form solution — we must use iterative gradient-based methods.

---

## The Loss Landscape

Imagine plotting $\mathcal{L}$ as a function of (two dimensions of) $W$:

```
Loss
 ▲
 │     ╭──╮      ╭──╮
 │    /    \    /    \
 │   /  ╭──╯    ╰──╮ \
 │──/───╯            ╰─\──► W
 │            ↑
 │        global min
 │
 │  Local minima, saddle points, flat regions — all make optimization hard
```

For a single linear classifier: convex (one global minimum). For deep networks: highly non-convex with millions of dimensions. But in high dimensions, true local minima (all eigenvalues of the Hessian positive) are exponentially rare — saddle points dominate instead.

---

## Gradient Descent

The gradient $\nabla_W \mathcal{L}$ points in the direction of steepest **ascent**. Step in the negative direction:

$$W \leftarrow W - \eta \nabla_W \mathcal{L}(W)$$

where $\eta$ is the **learning rate** (step size).

**Full batch gradient descent:** compute the gradient over the entire training set, then update. Expensive — for $N=1M$ images, one update requires $N$ forward passes.

**Stochastic Gradient Descent (SGD):** use a single random example per update. Cheap but very noisy.

**Mini-batch SGD:** use a random subset (batch) of $B=32$–$512$ examples per update. The standard in deep learning.

```python
def sgd(params, grads, lr=0.01):
    for p, g in zip(params, grads):
        p -= lr * g
```

**Why does mini-batch SGD work?**
- Gradient over mini-batch is an unbiased estimator of the full gradient
- Noise actually helps escape sharp minima (implicit regularization)
- GPU parallelism makes batch processing nearly free up to a point

---

## Momentum

Vanilla SGD oscillates in narrow valleys (large gradient in one direction, small in another). Momentum accumulates a velocity vector in directions of consistent gradient:

$$v_{t+1} = \mu v_t - \eta \nabla_W \mathcal{L}$$
$$W_{t+1} = W_t + v_{t+1}$$

$\mu = 0.9$ is typical. Think of a ball rolling downhill:

```
Without momentum:             With momentum (μ=0.9):
  loss                          loss
   ▲                             ▲
   │  ↙↗↙↗↙↗↙↗↗→→              │  →→→→→→→→→→→
   │  (oscillating)              │  (smooth descent)
   └─────────────► W             └─────────────► W
```

**Nesterov momentum:** compute the gradient at the "lookahead" position $W + \mu v$, not at $W$. Often faster convergence:

$$v_{t+1} = \mu v_t - \eta \nabla_W \mathcal{L}(W_t + \mu v_t)$$

---

## Adaptive Learning Rate Methods

Different parameters may need different learning rates. Weight matrices in deep layers vs shallow layers have very different gradient magnitudes.

### AdaGrad

Accumulate squared gradients; divide the learning rate by the root of the sum:

$$G_t = G_{t-1} + g_t^2$$
$$W \leftarrow W - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$$

Parameters with large historical gradients get small effective LR; rare parameters get large LR. Good for sparse features, but $G_t$ only grows → LR shrinks to zero → training stalls.

### RMSProp

Fix AdaGrad's stalling by using an **exponential moving average** of squared gradients:

$$v_t = \beta v_{t-1} + (1-\beta) g_t^2$$
$$W \leftarrow W - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t$$

$\beta = 0.99$ typical. The denominator adapts but doesn't monotonically increase.

### Adam: Adaptive Moment Estimation

Combines momentum (first moment) and RMSProp (second moment):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(first moment — momentum)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(second moment — RMSProp)}$$

**Bias correction:** at $t=0$, both $m_0=0$ and $v_0=0$, so early estimates are biased toward zero. Correct:

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**Update:**

$$W \leftarrow W - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

```python
def adam(params, grads, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    t += 1
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i] = b1 * m[i] + (1 - b1) * g
        v[i] = b2 * v[i] + (1 - b2) * g**2
        m_hat = m[i] / (1 - b1**t)
        v_hat = v[i] / (1 - b2**t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return t
```

Defaults: $\eta=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$. These work well across essentially all architectures and tasks.

**AdamW:** weight decay applied directly to weights (decoupled from the adaptive LR), not via gradient:

$$W \leftarrow W(1 - \eta\lambda) - \frac{\eta}{\sqrt{\hat{v}} + \epsilon}\hat{m}$$

AdamW is the standard in modern LLMs and vision transformers.

---

## Optimizer Comparison

```
               SGD     SGD+Mom   RMSProp    Adam
Convergence    Slow    Medium    Fast       Fast
Tuning         Easy    Medium    Hard       Easy
Adaptive LR    No      No        Yes        Yes
Works OOTB     No      No        Sometimes  Yes
Memory         1×      2×        2×         3×

General advice:
  Quick experiment / paper baseline  → Adam (lr=3e-4)
  Training from scratch, fine-tuned → SGD+Nesterov+LR schedule
  LLMs and ViTs                     → AdamW
```

---

## Learning Rate Schedules

The learning rate is the most important hyperparameter. Too large → diverges. Too small → converges to a bad minimum.

### Step decay

Drop LR by a factor every $k$ epochs:

```
LR
 ▲
 │───
 │   │
 │   └───
 │       │
 │       └─── ...
 └────────────────► epochs
```

### Cosine annealing

Smoothly decay following a half-cosine curve. Often paired with **warm restarts** (SGDR):

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$$

```
LR
 ▲
 │╮    ╮    ╮
 │ ╰╮  ╰╮  ╰╮
 │  ╰╮  ╰╮  ╰╮
 └────────────► iterations
```

### Linear warmup + cosine decay

The standard for transformers. Ramp LR from 0 to peak over the first few thousand steps, then cosine decay. Prevents instability from large random initial gradients.

---

## Second-Order Methods

Newton's method uses the Hessian $H = \nabla^2 \mathcal{L}$:

$$W \leftarrow W - H^{-1} \nabla \mathcal{L}$$

Naturally adapts step size to curvature — large steps in low-curvature directions, small in high-curvature. No LR tuning needed.

**Why not used in deep learning?** The Hessian has $D^2$ elements. For a 7B model, $D \approx 7 \times 10^9$ → Hessian has $5 \times 10^{19}$ elements. Completely infeasible. Adam is a rough first-order approximation to Newton's method (diagonal Hessian approximation via $\sqrt{v_t}$).
