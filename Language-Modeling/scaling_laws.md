# Scaling Laws

Scaling laws describe how model performance changes as we scale compute, data, and parameters. They enable principled decisions about where to invest resources — a crucial input to the multi-million-dollar decision of what model to train.

---

## The Kaplan scaling laws (OpenAI, 2020)

Kaplan et al. found that language model loss follows clean power laws in model size $N$, dataset size $D$, and compute $C$:

$$L(N) \sim N^{-\alpha}, \quad L(D) \sim D^{-\beta}, \quad L(C) \sim C^{-\gamma}$$

with roughly $\alpha \approx 0.076$, $\beta \approx 0.095$, $\gamma \approx 0.050$ (empirically fit).

**Key claim from Kaplan:** for a fixed compute budget $C$, model size matters much more than data. Optimal $N^* \propto C^{0.73}$, meaning data should scale more slowly: $D^* \propto C^{0.27}$.

This led to the intuition that you should **train large models on relatively little data** — which is what GPT-3 did.

---

## Chinchilla (Hoffmann et al., DeepMind, 2022)

Hoffmann et al. revisited these laws with a more careful experimental protocol, finding a different optimal allocation:

$$\text{Optimal: } N \propto C^{0.5}, \quad D \propto C^{0.5}$$

In plain English: **model size and token count should scale equally**. For every doubling of compute budget, double both the model and the data.

**The Chinchilla rule:** a well-trained model uses approximately **20 tokens per parameter**.

| Model | Params | Tokens used | Chinchilla-optimal tokens |
|-------|--------|-------------|--------------------------|
| GPT-3 | 175B | 300B | ~3.5T |
| Chinchilla | 70B | 1.4T | ~1.4T ✓ |
| LLaMA-1 | 65B | 1.4T | ~1.3T ✓ |
| LLaMA-2 | 70B | 2T | ~1.4T (slightly over) |

**Why does this matter?** Training a smaller, better-trained model is better for inference cost: a 70B model trained on 1.4T tokens (Chinchilla) outperforms a 280B model trained on 300B tokens (GPT-3 style), while being 4× cheaper to serve.

---

## The loss formula

Hoffmann et al. fit a combined loss function:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

where $E \approx 1.69$ is the irreducible entropy of the data, and $A, B, \alpha, \beta$ are fit to data.

For a fixed compute budget $C \approx 6ND$ (the 6× factor accounts for forward + backward):

$$\min_{N, D \text{ s.t. } 6ND=C} L(N, D)$$

This optimization yields the equal-scaling result.

---

## Inference-time compute scaling

Recent work (OpenAI o1, DeepSeek-R1) shows a complementary scaling axis: **more compute at inference time** can substitute for larger models.

$$L(\text{quality}) \sim C_{\text{train}}^{-\alpha} \approx C_{\text{inference}}^{-\beta}$$

The idea: let the model "think longer" (generate more tokens) before answering. A smaller model that reasons for 1000 tokens can match a larger model answering immediately.

This is the foundation of chain-of-thought and reasoning models (see [reasoning_models.md](reasoning_models.md)).

**Compute-optimal inference:** given a fixed inference budget, what's the best model size and how many tokens should it generate? Recent work suggests the optimal inference strategy uses a model smaller than you might expect, generating many thinking tokens.

---

## Emergent abilities

Some capabilities appear abruptly as models scale, showing near-zero performance below a threshold and then rapid improvement:

- Few-shot learning (GPT-3, ~50B params)
- Chain-of-thought reasoning (~100B params)
- Instruction following without fine-tuning
- Arithmetic with large numbers

**Are emergence effects real?** Wei et al. (2022) documented many emergent abilities. Schaeffer et al. (2023) argued these are **artifacts of the evaluation metric** — discontinuous metrics (like exact match accuracy on arithmetic) hide gradual underlying improvements that would look smooth under continuous metrics (like token-level accuracy).

The debate: true phase transitions in capability vs. measurement artifacts. Both are probably true for different abilities.

---

## Scaling beyond loss: benchmark performance

Loss is a proxy metric. Downstream task performance often scales differently:

- Some tasks scale smoothly with loss (natural language generation, summarization)
- Some tasks show emergent jumps (multi-step reasoning, code)
- Some tasks plateau early (factual recall bounded by training data)
- Some tasks show U-shaped scaling (very large models can be worse on some tasks)

**The bitter lesson:** it's hard to predict which tasks benefit from scale before actually training. Benchmark prediction remains an open problem.

---

## Scaling with data quality

Recent work shows that **data quality can substitute for scale**: a smaller model trained on cleaner data can match a larger model trained on noisy data.

The Phi family (Microsoft) trains 1–3B models on synthetic "textbook-quality" data and matches much larger models on reasoning benchmarks. This challenges the "data is cheap, scale up parameters" narrative.

$$L(N, D, Q) \approx E + \frac{A}{N^\alpha} + \frac{B}{(D \cdot Q)^\beta}$$

where $Q$ is a data quality multiplier.

---

## Practical implications for model design

1. **Budget-constrained training:** use Chinchilla's 20 tokens/param rule to decide model size
2. **Inference-constrained deployment:** train a smaller model on more tokens (LLaMA philosophy) — cheap at inference, might need more training compute
3. **Data-limited regime:** if you don't have enough tokens (specialized domains), scale model slowly and focus on data quality
4. **Multiple epochs:** Chinchilla assumes one epoch. Training multiple epochs is possible but shows diminishing returns after ~4 epochs (Taylor et al., 2024)
