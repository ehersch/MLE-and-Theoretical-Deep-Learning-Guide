# Midtraining (Continued Pretraining)

Midtraining refers to a training phase between pretraining and post-training. The base pretrained model has general capabilities; midtraining specializes it for a domain, extends its context window, or injects new knowledge — without the full cost of pretraining from scratch.

---

## Why midtraining?

**Pretraining** gives a model general language understanding but may not have:
- Enough code data (if you want a coding model)
- Domain-specific knowledge (medical, legal, financial)
- Long-context capability (if pretrained at 4k)
- Recent knowledge (if training data has a cutoff)

**Fine-tuning alone** can adapt behavior but can't reliably inject large amounts of new factual knowledge — the model may hallucinate when asked about facts outside the pretraining distribution.

**Midtraining** (continued pretraining on domain-specific data at full LM loss) is the solution.

---

## Continued pretraining on domain data

Start from a pretrained checkpoint and continue pretraining on domain-specific data, optionally mixed with general data.

**Key design choices:**

### Data mixing

Pure domain data risks **catastrophic forgetting** — the model forgets general capabilities. Mix domain data with some general data:

$$\mathcal{L} = (1 - \lambda) \mathcal{L}_{\text{domain}} + \lambda \mathcal{L}_{\text{general}}$$

Typical $\lambda$ ranges from 0.1 to 0.5 depending on how different the domain is. Too much general data slows domain adaptation; too little causes forgetting.

### Learning rate

Restart from a small learning rate (e.g., 10× smaller than peak pretraining LR). The model's weights are already good; large LRs can destroy them.

Some practitioners use a brief warmup then cosine decay for the continued pretraining phase.

### Epoch concerns

Pretraining data is usually seen once. Domain data may be small enough to require multiple epochs. After ~4 epochs, performance typically stops improving and may degrade (the model memorizes rather than generalizes).

---

## Context length extension

Most models pretrain at 4k or 8k context. Extending to 128k+ requires midtraining.

### Why doesn't the pretrained model handle long context?

Positional encodings (especially RoPE) are only trained on positions up to the pretraining context length. Attending to positions 50k+ produces out-of-distribution position embeddings → garbage attention patterns.

### NTK-aware interpolation

The idea: "stretch" the RoPE frequencies so that positions within the original context map to longer effective positions:

$$\theta'_i = \theta_i \cdot \left(\frac{L'}{L}\right)^{-2i/d}$$

where $L'$ is the target context length, $L$ is the original. This rescales the base of the rotary embedding so higher positions are still "in-distribution."

**YaRN (Yet another RoPE extensioN):** combines NTK interpolation with attention temperature scaling, achieving near-full quality at 128k context with only a small midtraining phase (~400M tokens).

### Long-context midtraining recipe

1. Apply NTK/YaRN RoPE scaling to the model
2. Continue pretraining on **long documents** (papers, books, code repos) — regular web text is mostly <1k tokens, so a data mix shift toward long documents is needed
3. Train for a relatively small number of tokens (1–5% of original pretraining) to adapt to the new position scale

---

## Knowledge injection

Adding new facts (post-2023 events, private data) is notoriously hard. Options:

**Continued pretraining:** works, but requires many exposures of a fact (models need ~10–100 repetitions to reliably recall a new fact) and can still hallucinate.

**RAG (retrieval-augmented generation):** retrieve facts at inference time rather than baking them in. Generally better for dynamic, frequently-updated knowledge (see [agents_and_rag.md](agents_and_rag.md)).

**Hybrid:** midtrain on new knowledge corpus + use RAG for precise retrieval. The midtrained model is better at reasoning about the domain; RAG ensures specific facts are accurate.

---

## Code and math midtraining

Models pretrained predominantly on text often underperform on code and math reasoning.

**CodeLLaMA:** LLaMA 2 → midtraining on 500B tokens of code → further long-context midtraining → instruction-tuned.

**DeepSeek-Math:** 7B model midtrained on 120B tokens of math-specific data achieves significantly better math reasoning than the base model.

**Key observations:**
- Code midtraining incidentally improves reasoning on non-code tasks (code is structured, logical)
- Math data benefits from including step-by-step solutions (not just problems + answers)
- Synthetic data (LLM-generated worked solutions) is valuable when human data is scarce

---

## Instruct data during midtraining (blending)

Some labs blend instruction-following data into the midtraining phase (not just the posttraining phase). This is sometimes called **supervised fine-tuning at pretraining scale**.

Benefits:
- The model is larger and more capable when it first sees instruction data
- Instruction formatting becomes part of the base model's distribution

Risks:
- Instruction data has very different formatting from raw text → may confuse the model if ratio is too high
- Reduces raw pretraining efficiency

Typical ratio: ~1–5% instruction data by token count in the midtraining phase.
