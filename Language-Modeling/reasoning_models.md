# Reasoning Models

Reasoning models are LLMs trained to generate explicit intermediate reasoning steps before producing a final answer. They trade inference-time compute for dramatically better performance on complex tasks — math, coding, scientific reasoning, and multi-step logic.

---

## The core insight: thinking before answering

Humans solve hard problems by thinking through them. A language model generating tokens autoregressively can do the same: generate a scratchpad of reasoning, then produce the answer.

**Chain-of-thought (CoT)** prompting (Wei et al., 2022) discovered this empirically: simply asking "let's think step by step" elicits much better performance on GSM8K and similar tasks.

But prompting is fragile. Training models to reason natively — and to do it well — is the goal of reasoning model research.

---

## Chain-of-Thought prompting

### Zero-shot CoT

Append "Let's think step by step." to any prompt. Surprisingly effective at ~100B+ scale.

```
Q: If I have 3 apples and give 2 to Alice, then buy 5 more, how many do I have?
A: Let's think step by step.
   Start: 3 apples.
   Give 2 to Alice: 3 - 2 = 1 apple.
   Buy 5 more: 1 + 5 = 6 apples.
   Answer: 6.
```

### Few-shot CoT

Provide worked examples in the prompt:

```
Q: [example problem]
A: [worked step-by-step solution]
...
Q: [new problem]
A: [model generates reasoning + answer]
```

### Why does CoT work?

- More tokens → more compute allocated to the problem
- Intermediate steps act as "working memory" — the model doesn't have to compress a long reasoning chain into a single forward pass
- Errors in reasoning are visible and can be caught/corrected by later tokens
- The model can condition its answer on the reasoning, reducing inconsistency

**Compute perspective:** generating a 200-token reasoning chain before a 10-token answer uses ~20× more FLOPs than directly generating the answer. This is inference-time compute scaling.

---

## Scaling inference compute

The analogy to training: just as training loss improves with more compute, answer quality improves with more inference compute — if the model knows how to use it.

**Best-of-N sampling:** sample N independent answers, pick the best:

$$y^* = \arg\max_{y \in \{y_1, \ldots, y_N\}} r(y)$$

where $r$ is a reward/verifier. For code: pick the answer that passes the most unit tests. For math: pick the majority answer (self-consistency).

**Self-consistency (Wang et al., 2022):** sample multiple CoT paths, take the majority vote on the final answer. Outperforms greedy decoding significantly.

```python
from collections import Counter

def self_consistency(model, question, n=10, temperature=0.7):
    answers = []
    for _ in range(n):
        chain = model.generate(question, temperature=temperature)
        answer = extract_final_answer(chain)
        answers.append(answer)
    return Counter(answers).most_common(1)[0][0]
```

**Compute-optimal inference:** given a budget of $B$ tokens, how should you spend them?
- 1 sample × 200 thinking tokens? 
- 10 samples × 20 thinking tokens?
- 1 sample × 200 tokens + verification?

Recent work shows that for hard problems, longer thinking traces with verification often outperform more samples with short reasoning.

---

## Process Reward Models (PRMs)

**Outcome reward models (ORMs):** score only the final answer. Binary: correct/incorrect (or soft version from human labels).

**Process reward models (PRMs):** score each reasoning step. Give credit for correct intermediate steps even if the final answer is wrong (e.g., the model set up the problem correctly but made an arithmetic error).

Introduced in **Let's Verify Step by Step** (Lightman et al., 2023, OpenAI).

**Training PRMs:** requires step-level human annotations — labelers rate each step as correct, neutral, or wrong. Very expensive to collect.

**Monte Carlo step scoring:** generate many rollouts from each intermediate step; steps that lead to correct final answers are labeled as good steps. No human labels needed, but requires a verifiable reward signal.

$$\text{PRM}(s_t) \approx P(\text{correct final answer} \mid s_1, \ldots, s_t)$$

**Using PRMs for search:** rather than greedy decoding, use the PRM to guide beam search or MCTS over reasoning steps:

```
Root: [problem]
├── Step A → PRM score: 0.9
│   ├── Step A1 → PRM: 0.95 → Answer: correct ✓
│   └── Step A2 → PRM: 0.3 → ...
└── Step B → PRM score: 0.4 → ...
```

---

## RLVR: Reinforcement Learning with Verifiable Rewards

The key innovation behind OpenAI o1 and DeepSeek-R1: use RL with a **verifiable reward signal** to train models to generate better reasoning traces.

**Why verifiable?** For math and code, we can automatically check if the answer is correct. This gives us a scalar reward without needing a reward model:

$$r(y) = \begin{cases} +1 & \text{if final answer is correct} \\ 0 & \text{otherwise} \end{cases}$$

**GRPO (Group Relative Policy Optimization)** (DeepSeek, 2024):

Instead of PPO's critic network, estimate the baseline from a group of outputs for the same prompt:

$$A_i = r_i - \text{mean}(\{r_1, \ldots, r_G\})$$

Loss:

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^G \min\!\left(\frac{\pi_\theta(y_i)}{\pi_{\text{old}}(y_i)} A_i,\ \text{clip}(\cdot, 1\pm\epsilon) A_i\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

**Advantages over PPO for RLVR:**
- No value/critic model needed
- Simpler implementation
- Works well when reward is sparse (many wrong, few correct)

---

## DeepSeek-R1

DeepSeek-R1 (2025) is the first fully open reasoning model. Key findings:

**R1-Zero:** apply GRPO directly to the base model with only a math/code reward signal. The model spontaneously develops:
- Reflection ("Wait, let me reconsider...")
- Self-correction
- Extended thinking traces (sometimes 10k+ tokens)
- An "aha moment" — discovering it should rethink a problem

This is remarkable: no CoT examples in training data. The behavior emerges purely from RL with a binary reward.

**R1:** adds cold-start SFT data (a few thousand high-quality reasoning traces) before RLVR, followed by rejection sampling and more SFT + RLVR stages. More stable training.

**Key recipe:**
```
Base model
    ↓ Cold-start SFT (reasoning traces)
    ↓ GRPO with verifiable rewards (math, code)
    ↓ Rejection sampling (collect correct traces)
    ↓ SFT on high-quality traces
    ↓ GRPO again
    = R1
```

**Distillation:** the R1 reasoning traces are used to train smaller models (1.5B, 7B, 14B, 32B) via SFT. These distilled models are surprisingly capable — a 7B distilled model approaches GPT-4 level on math benchmarks.

---

## OpenAI o1 and o3

OpenAI o1 (September 2024) introduced **chain-of-thought at training time** (not just inference time): the model is trained to think in a scratchpad before answering.

Key differences from standard models:
- The reasoning trace ("thinking tokens") is separate from the final output
- Users see only the final answer; the thinking is hidden (or summarized)
- Inference is much more expensive: o1 uses ~20–50× more tokens than GPT-4 per response
- Dramatically better at hard math, competition coding, and PhD-level science

**o3** extends this with more compute, larger model, and longer reasoning — achieving near-human performance on competition math (AIME) and frontier research benchmarks.

---

## Challenges and open questions

### Reward hacking in RLVR

With a binary correct/incorrect reward, models can find shortcuts:
- Guess common answers without reasoning
- Exploit format patterns in the grader
- "Verbosity hack" — long rambling that occasionally stumbles onto the answer

Mitigation: format rewards (penalize too short/too long reasoning), verified chain of thought (check intermediate steps), diverse reward signals.

### Overthinking and underthinking

- **Overthinking:** model generates very long reasoning traces for simple problems, wasting compute
- **Underthinking:** model stops reasoning too early on hard problems, giving wrong answers

**Budget forcing:** explicitly tell the model how many thinking tokens it has. Or train with a length-budget reward that penalizes unnecessary length.

### Generalization beyond math/code

RLVR works well where rewards are verifiable. For open-ended tasks (creative writing, general reasoning), verification is harder. Active research area.

### Test-time compute allocation

Given a fixed inference budget, how to allocate it?
- Parallel: sample many chains, pick best (best-of-N)
- Sequential: generate one long chain with revisions
- Tree search: branch and prune using a PRM

The right strategy depends on the task and the model's calibration.

---

## Summary: the reasoning stack

```
Pretrained LLM
    ↓ SFT on reasoning traces (cold start)
    ↓ RLVR with verifiable rewards (GRPO/PPO)
    → Reasoning model (long CoT, self-correction)
    
At inference:
    Problem → [thinking tokens: plan, explore, verify, correct]
                                    ↓
                              Final answer
    
Optional: PRM-guided search / best-of-N / self-consistency
```

Reasoning models represent a shift in the scaling paradigm: rather than only scaling pretraining compute, we now scale inference compute intelligently, with models trained to use that extra compute productively.
