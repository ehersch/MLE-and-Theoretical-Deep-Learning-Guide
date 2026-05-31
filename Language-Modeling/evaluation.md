# Evaluation

Evaluating LLMs is hard. A model can score well on a benchmark while failing at real-world tasks, or vice versa. Good evaluation requires understanding what each metric actually measures and where it breaks down.

---

## Perplexity

The most fundamental LLM metric. For a test sequence $x_1, \ldots, x_T$:

$$\text{PPL} = \exp\!\left(-\frac{1}{T} \sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})\right)$$

Perplexity is the exponentiated average negative log-likelihood per token. Intuitively: the geometric mean of the inverse probability assigned to each token.

- **PPL = 1:** perfect, assigns probability 1 to every token (impossible on real data)
- **PPL = V (vocab size):** random model
- **State-of-the-art (2024):** ~5–8 PPL on test sets like WikiText-103

**When to use:** comparing models of the same architecture trained on the same data. Perplexity is not comparable across different tokenizers (longer tokenizations inflate perplexity).

**Limitations:**
- Measures average likelihood, not specific capabilities
- A model can have low PPL but fail on reasoning tasks
- Sensitive to tokenization: the same text tokenized differently gives different PPL

```python
def perplexity(model, tokens):
    with torch.no_grad():
        logits = model(tokens[:-1])  # (T-1, vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs[range(len(tokens)-1), tokens[1:]].mean()
    return torch.exp(nll).item()
```

---

## Benchmark suites

### MMLU (Massive Multitask Language Understanding)

57 subjects (math, science, history, law, medicine, etc.), multiple choice (4 options). Tests broad knowledge.

**Format:**
```
Question: Which of the following is NOT a property of a normal distribution?
A) Mean = Median = Mode
B) It is always positively skewed
C) 68% of data within 1 std dev
D) It is symmetric
Answer: B
```

**Evaluation:** few-shot (5-shot standard), measure accuracy. GPT-4: ~87%, LLaMA-3-8B: ~68%.

**Limitations:** MC format doesn't test generation; data contamination is a serious concern (many benchmark questions appear in web crawl training data).

### HellaSwag

Commonsense NLI: given a partial description of a physical scenario, pick the most plausible continuation. Adversarially constructed to fool previous models.

### ARC (AI2 Reasoning Challenge)

Science questions; ARC-Challenge subset is hard (requires reasoning, not just retrieval).

### GSM8K / MATH

Grade-school math word problems (GSM8K, 8k problems) and competition math (MATH, 12k problems). Requires multi-step arithmetic and symbolic reasoning.

State-of-the-art on GSM8K: >95% (GPT-4, Claude 3.5). MATH: ~70% (much harder).

### HumanEval / MBPP

Coding benchmarks: generate code that passes unit tests. HumanEval has 164 problems; MBPP has 374.

**pass@k:** probability that at least one of $k$ samples passes all tests.

### BIG-Bench Hard

Challenging subset of BIG-Bench tasks where GPT-4/PaLM still struggle: logical deduction, causal reasoning, word sorting.

### HELM (Holistic Evaluation of Language Models)

Stanford framework measuring multiple dimensions: accuracy, calibration, robustness, fairness, efficiency, disinformation susceptibility. More holistic than single-number rankings.

---

## LLM-as-judge

Automated evaluation using another LLM (typically GPT-4 or Claude) to score responses.

**Pairwise comparison:**
```
Judge prompt:
Given the following question: {question}
Response A: {response_a}
Response B: {response_b}
Which response is better? Explain your reasoning, then answer A, B, or Tie.
```

**Absolute scoring:**
```
Rate the following response on helpfulness (1-10), accuracy (1-10), 
clarity (1-10). Explain each score.
```

**Advantages:**
- Scales to thousands of examples
- Can evaluate open-ended generation (not just MC)
- Captures nuances missed by word overlap metrics

**Biases and failure modes:**
- **Position bias:** judges prefer the first or second response regardless of quality
- **Verbosity bias:** judges prefer longer responses (even when less informative)
- **Self-preference:** a model judging its own outputs inflates scores
- **Sycophancy:** judges agree with confident-sounding responses even if wrong

**Mitigations:** randomize A/B order, calibrate on human judgments, use multiple independent judges, use chain-of-thought before scoring.

### MT-Bench and Chatbot Arena

**MT-Bench:** 80 multi-turn questions across 8 categories, evaluated by GPT-4. Standard for comparing instruction-tuned models.

**Chatbot Arena (LMSYS):** humans compare anonymous model pairs. Outputs Elo ratings. Most naturalistic evaluation since real users ask real questions. Considered the gold standard for model ranking.

---

## Calibration

A well-calibrated model's confidence equals its actual accuracy. A model that says "I'm 80% confident" should be right 80% of the time.

**Expected Calibration Error (ECE):**

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

where $B_m$ is the $m$-th confidence bin.

**Overconfidence:** RLHF'd models tend to be overconfident and express uncertainty less than pretrained models. This is a known tension between helpfulness and honest calibration.

**Reliability diagrams:** plot confidence vs. accuracy. A perfectly calibrated model falls on the diagonal.

---

## Contamination

Benchmark data often appears verbatim in training corpora (Common Crawl contains much of the web, including benchmark datasets).

**Consequences:** models can appear to score higher than their actual capability.

**Detection methods:**
- n-gram overlap between training data and benchmark
- Performance gap between contaminated vs. de-duplicated versions
- Models sometimes "pattern-match" answers without understanding

**Mitigations:**
- Use held-out, recently-created benchmarks
- Dynamic evaluation (generate new examples at test time)
- Membership inference attacks to detect contamination

---

## Beyond accuracy: other dimensions

| Dimension | How to measure |
|-----------|----------------|
| Faithfulness | Does generated text match source? (for RAG, summarization) |
| Robustness | Accuracy under input perturbations (typos, paraphrases) |
| Consistency | Same question → same answer across runs/phrasings |
| Safety | Rate of harmful outputs on adversarial prompts |
| Latency | Time to first token, tokens/second |
| Instruction following | Compliance with format constraints |

---

## Practical advice: what to evaluate

For a new model or fine-tune, a minimal eval suite:
1. **Perplexity** on a held-out slice of your domain data
2. **MMLU** (5-shot) — quick sanity check on general knowledge
3. **GSM8K** — math reasoning
4. **HumanEval** — if coding matters
5. **MT-Bench or human evals** — final quality check on open-ended generation
6. **Safety evals** — before any deployment
