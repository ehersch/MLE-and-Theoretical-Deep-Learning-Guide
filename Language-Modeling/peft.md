# Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning updates all model parameters — expensive in compute and memory, and can cause catastrophic forgetting. PEFT methods adapt large models by updating only a small subset of parameters while keeping the pretrained weights frozen.

---

## Why PEFT?

A 7B model has 7 billion parameters. Full fine-tuning in FP16 needs:
- Model: 14 GB
- Gradients: 14 GB
- Optimizer states (Adam): 56 GB
- **Total: ~84 GB**

With LoRA (rank 16), you update ~0.1% of parameters:
- LoRA parameters: ~20M params = 0.4 GB
- Frozen model (inference only): 14 GB
- **Total: ~15 GB** — fits on a single consumer GPU

PEFT also mitigates **catastrophic forgetting**: since base weights are frozen, the model retains general capabilities while adapting to the new task.

---

## Adapter layers

Early PEFT approach: insert small trainable feed-forward modules after each transformer sub-layer.

```
x → [Frozen transformer layer] → down_proj (d → r) → activation → up_proj (r → d) → + x
```

Typically $r \ll d$ (e.g., $r=64$, $d=4096$). Only adapter weights are trained.

**Downside:** adds latency at inference (extra forward passes through adapters).

---

## LoRA: Low-Rank Adaptation

Hu et al. (2021). The key insight: weight updates during fine-tuning have low intrinsic rank. Rather than learning $\Delta W \in \mathbb{R}^{d \times k}$ directly, constrain it to be a product of two low-rank matrices:

$$W' = W + \Delta W = W + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

At initialization: $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$, so $\Delta W = 0$ and training starts from the pretrained model.

The forward pass:

$$h = Wx + \frac{\alpha}{r} BAx$$

The $\alpha/r$ scaling factor: $\alpha$ is a hyperparameter (often set to $r$ or $2r$), and dividing by $r$ ensures the update scale is independent of $r$.

**Parameter count:** $r(d + k)$ vs $dk$ for full fine-tuning. For $d=k=4096$, $r=16$: 131K vs 16M params (99.2% reduction).

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear, rank=16, alpha=16):
        super().__init__()
        d, k = linear.weight.shape
        self.linear = linear  # frozen
        self.A = nn.Parameter(torch.randn(rank, k) * 0.01)
        self.B = nn.Parameter(torch.zeros(d, rank))
        self.scale = alpha / rank
        
        # Freeze base weights
        for p in self.linear.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        return self.linear(x) + self.scale * (x @ self.A.T @ self.B.T)
```

### Which matrices to LoRA?

In transformers: $W_Q, W_K, W_V, W_O$ (attention) and $W_1, W_2$ (FFN). Original LoRA applied only to $W_Q, W_V$. Applying to all weight matrices generally works better for instruction fine-tuning.

### Merging LoRA at inference

After training, merge $\Delta W = BA$ into $W$: $W' = W + BA$. No inference overhead.

```python
# Merge LoRA into base weights
W_merged = base_linear.weight + lora.B @ lora.A * lora.scale
```

---

## QLoRA: Quantized LoRA

Dettmers et al. (2023). Fine-tune a quantized (4-bit) base model with LoRA adapters in BF16.

**Three innovations:**
1. **NF4 quantization** (see [quantization.md](quantization.md)): base model in 4-bit NF4, nearly lossless
2. **Double quantization:** quantize the quantization constants themselves (saves ~0.4 bits/param)
3. **Paged optimizers:** use CUDA unified memory to handle gradient checkpointing spikes without OOM

**Memory:** 4-bit base + BF16 LoRA adapters:
- 7B model: ~6 GB (4-bit) + ~0.5 GB (LoRA) = **~7 GB** — fits on consumer GPUs

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained("model", quantization_config=bnb_config)

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

---

## Prompt tuning

Prepend a small number of soft (continuous) "prefix" tokens to the input. Only these prefix tokens are trained; the model is fully frozen.

**Prompt tuning:** soft tokens prepended at the input embedding layer only:

$$h_0 = [\underbrace{p_1, \ldots, p_k}_{\text{trainable}}, \text{embed}(x_1), \ldots, \text{embed}(x_T)]$$

**Prefix tuning** (Li & Liang, 2021): soft prefixes at every layer's K and V:

$$K_l = [P_l^K; W_K^l X], \quad V_l = [P_l^V; W_V^l X]$$

Each layer has its own trainable prefix parameters, giving more capacity.

**Tradeoffs vs LoRA:**
- Fewer parameters (just prefix tokens)
- But: hard to optimize (gradients must flow through frozen layers back to prefix)
- Inference: prefix tokens consume context window
- Quality: generally worse than LoRA for complex tasks

---

## LoRA variants

### LoRA+
Different learning rates for $A$ and $B$ matrices (set $B$'s LR higher). Empirically better.

### DoRA (Weight-Decomposed LoRA)
Decompose weight updates into magnitude and direction, applying LoRA only to the directional component. Better quality than vanilla LoRA.

### VeRA (Vector-based Random Matrix Adaptation)
Share random frozen $A$ and $B$ across all layers; train only small per-layer scaling vectors. Extreme parameter efficiency.

### LoRA-FA
Freeze $A$, train only $B$. Reduces memory for storing $A$'s gradient; minor quality degradation.

---

## Choosing rank $r$

- **$r=4$:** very parameter-efficient, good for simple style adaptation
- **$r=16$:** standard for instruction following
- **$r=64$:** approaching full fine-tuning quality for complex tasks
- **$r=256+$:** nearly full fine-tuning; use when you have enough data and want max quality

**Rule of thumb:** start with $r=16$, tune if needed. Increasing $r$ beyond 64 shows diminishing returns for most tasks.

---

## PEFT for instruction following vs. specialized tasks

| Task | Recommended | Notes |
|------|-------------|-------|
| General instruction following | LoRA $r=16$, all attention | Large, diverse data |
| Domain fine-tuning | LoRA $r=64$ on all weights | May need more capacity |
| Consumer hardware | QLoRA, $r=16$ | 4-bit base, single GPU |
| Style/persona | Prompt tuning | Few-shot level data |
| Multi-task | Separate adapters per task, shared base | Modular composition |
