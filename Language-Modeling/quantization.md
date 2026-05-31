# Quantization

Quantization reduces the numerical precision of model weights (and sometimes activations) from FP16/BF16 to INT8, INT4, or lower. This shrinks model size, reduces memory bandwidth requirements, and can speed up inference — often with surprisingly little quality degradation.

---

## Why quantize?

A 70B model in BF16:
- Weights: $70 \times 10^9 \times 2$ bytes = **140 GB** → needs 2+ A100s just to load
- At INT4: **35 GB** → fits on a single A100

Beyond storage: inference is often memory-bandwidth-bound (see [systems_and_hardware.md](systems_and_hardware.md)). Halving weight size → halving HBM reads → ~2× faster decode throughput.

---

## Quantization fundamentals

### Uniform quantization

Map a floating-point range $[x_{\min}, x_{\max}]$ to $n$-bit integers $[0, 2^n - 1]$:

$$\hat{x} = \text{round}\!\left(\frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (2^n - 1)\right)$$

Dequantize back:

$$x \approx \hat{x} \cdot \frac{x_{\max} - x_{\min}}{2^n - 1} + x_{\min}$$

This is equivalently written as $x \approx s \cdot \hat{x} + z$, where $s$ is the **scale** and $z$ is the **zero point**.

**Symmetric quantization:** $z=0$, $s = x_{\max} / (2^{n-1} - 1)$. Simpler, preferred for weights.

**Asymmetric:** $z \neq 0$. Better for activations which are often non-symmetric (after ReLU/SiLU).

### Granularity

- **Per-tensor:** one scale for the whole weight matrix. Fastest, least precise.
- **Per-channel (per-row/column):** one scale per output channel. Better quality, minor overhead.
- **Group quantization:** one scale per group of $g$ elements (e.g., $g=128$). Balance between quality and overhead. Used in GPTQ, AWQ.

---

## Post-Training Quantization (PTQ)

Quantize a trained model without any further training. Calibrate the scale using a small dataset.

### Naive round-to-nearest (RTN)

Just round each weight to the nearest quantized value. Fast, but can degrade significantly below INT8.

```python
def quantize_tensor(W, bits=8, group_size=128):
    W = W.reshape(-1, group_size)
    scale = W.abs().max(dim=-1, keepdim=True).values / (2**(bits-1) - 1)
    W_quant = (W / scale).round().clamp(-(2**(bits-1)), 2**(bits-1)-1)
    return W_quant.to(torch.int8), scale
```

### GPTQ (Frantar et al., 2022)

Layer-by-layer quantization that minimizes the **reconstruction error** for each linear layer:

$$\min_{\hat{W}} \|WX - \hat{W}X\|_F^2$$

where $X$ is activations from a calibration dataset.

GPTQ uses a second-order method: the Hessian $H = XX^\top$ tells us which weights matter most. Weights are quantized one at a time, with the remaining unquantized weights adjusted to compensate:

$$\delta w_j = -\frac{w_j - Q(w_j)}{[H^{-1}]_{jj}} \cdot H^{-1}_{:,j}$$

This is the OBQ (Optimal Brain Quantization) framework. GPTQ makes it practical at scale with batched processing and Cholesky decompositions.

**Result:** INT4 GPTQ at 4 bits/weight with minimal perplexity degradation (<1 PPL increase for 7B+ models).

### AWQ (Lin et al., 2023)

Activation-aware quantization. Observation: not all weights are equally important — weights that multiply large activations cause larger quantization errors.

**Key insight:** protect the ~1% of weights corresponding to salient channels by scaling them up before quantization:

$$Q(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1}$$

The scale $s$ is found by searching for values that minimize quantization error on a calibration set. Because $s$ is absorbed into adjacent weights, no runtime overhead.

AWQ often outperforms GPTQ at the same bit-width, especially at 4 bits.

---

## Quantization-Aware Training (QAT)

Train with simulated quantization in the forward pass (fake-quantize), so the model adapts to quantization noise:

```python
def fake_quantize(x, scale, bits=8):
    x_q = (x / scale).round().clamp(-128, 127)
    return x_q * scale  # straight-through estimator for backward

# During training:
y = fake_quantize(W, scale) @ x
loss = criterion(y, target)
loss.backward()  # gradient flows through fake_quantize as if it's identity
```

The **straight-through estimator (STE)** passes gradients through the non-differentiable rounding operation.

QAT recovers quality lost from PTQ, especially at aggressive bit widths (<4 bits), but requires full training runs — expensive for large models.

---

## NF4: NormalFloat 4-bit (QLoRA)

Standard INT4 assumes uniform distribution of values. Model weights are approximately normally distributed. **NF4** defines quantization levels at the quantiles of a normal distribution, minimizing expected quantization error:

$$q_i = \Phi^{-1}\!\left(\frac{i}{2^k - 1}\right), \quad i = 0, \ldots, 2^k - 1$$

where $\Phi^{-1}$ is the inverse normal CDF.

The 16 NF4 levels are: `[-1.0, -0.6962, -0.5251, ..., 0.5251, 0.6962, 1.0]`

Each weight is stored as a 4-bit index into this lookup table + a per-group scale (in BF16). This is **not** hardware-native INT4 — dequantization happens on the fly. Used in QLoRA for fine-tuning quantized models.

```python
# Simplified NF4 quantization
nf4_table = [-1.0, -0.6962, -0.5251, -0.3949, -0.2840, -0.1848, -0.0911, 0.0,
              0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]

def quantize_nf4(w, group_size=64):
    w = w.reshape(-1, group_size)
    scale = w.abs().max(dim=-1, keepdim=True).values
    w_norm = w / scale
    # Find nearest NF4 level for each element
    dists = (w_norm.unsqueeze(-1) - torch.tensor(nf4_table)).abs()
    indices = dists.argmin(dim=-1).to(torch.uint8)  # 4-bit indices
    return indices, scale
```

---

## INT8 weight + FP16 activation (W8A16)

The most common production quantization scheme:
- Weights stored as INT8
- Dequantized to FP16 on the fly before matmul
- Activations remain FP16

**LLM.int8()** (Dettmers et al., 2022): decomposed quantization that handles outlier activation channels in FP16, quantizes the rest in INT8. Achieves near-FP16 quality.

```python
import bitsandbytes as bnb
model = AutoModelForCausalLM.from_pretrained("model", load_in_8bit=True)
```

---

## W4A8 and W4A4

Emerging for next-generation inference:
- **W4A8:** 4-bit weights, 8-bit activations. Requires INT4 multiply-accumulate in hardware. Supported on H100.
- **W4A4:** 4-bit both. Very aggressive; quality degrades without careful handling of outliers.

---

## Practical tradeoffs

| Method | Bits | Quality | Speed | Cost |
|--------|------|---------|-------|------|
| FP16 baseline | 16 | ★★★★★ | 1× | high memory |
| LLM.int8() | 8 | ★★★★★ | ~1× (overhead) | 2× less memory |
| GPTQ INT4 | 4 | ★★★★☆ | 2× faster decode | 4× less memory |
| AWQ INT4 | 4 | ★★★★☆ | 2× faster decode | 4× less memory |
| NF4 (QLoRA) | 4 | ★★★★☆ | slower (lookup) | 4× less memory |
| 2-bit | 2 | ★★★☆☆ | — | experimental |

**General rule:** INT8 is almost free in quality; INT4 costs ~0.5–1 PPL; INT2 is risky without careful technique.
