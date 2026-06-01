# Model Distillation

**Knowledge distillation** transfers capability from a large, expensive **teacher** model into a smaller, cheaper **student** model. The student doesn't just learn from ground-truth labels — it learns from the teacher's *distribution* over outputs, which carries far more information.

---

## Why Distillation?

Training a 70B model costs millions of dollars. Serving one costs orders of magnitude more than serving a 7B model. But a naive 7B model trained from scratch may not be as good as a 7B model that has *learned from* a 70B model.

The key insight: **a teacher's output distribution is a richer training signal than a one-hot label.**

```
Ground truth label:   [0, 0, 0, 1, 0, 0, 0]   ← one bit of information

Teacher's softmax:    [0.001, 0.003, 0.002, 0.94, 0.03, 0.02, 0.004]
                              ↑
                  Tells the student WHICH wrong answers are plausible,
                  which concepts are related, the geometry of the space.
                  Much more information per example.
```

This was Hinton et al.'s original insight (2015): use the teacher's **soft targets** instead of hard labels. The soft targets encode the teacher's understanding of similarity between classes.

---

## The Three Paradigms of Post-Training

Before going deep on distillation, it's worth seeing where it fits in the landscape of post-training methods:

```
                    Data source       Supervision signal
                    ───────────────────────────────────
SFT                 Fixed dataset     Dense (one label per token)
                    (off-policy)      but off-policy — student never
                                      generated these sequences

RL (PPO/GRPO)       Student rollouts  Sparse (one score per episode)
                    (on-policy)       but on-policy — student generated
                                      these sequences

On-Policy           Student rollouts  Dense (one signal per token)
Distillation        (on-policy)       AND on-policy ← best of both worlds
```

The key trade-off in the first two:
- **SFT** has dense supervision (gradient at every token) but the student is evaluated on sequences it never generated. At test time it generates different sequences → distribution shift.
- **RL** is on-policy (student generates its own data) but reward is sparse: one number at the end of a full response.

On-policy distillation gets both: the student generates the sequences, and the teacher provides a dense signal at every token.

---

## Standard (Offline) Knowledge Distillation

The classic setup for classification:

$$\mathcal{L}_{\text{KD}} = (1 - \alpha)\, \mathcal{L}_{\text{CE}}(y, p_S) + \alpha\, \tau^2\, D_{\text{KL}}(p_T^\tau \| p_S^\tau)$$

- $p_T^\tau = \text{softmax}(\text{logits}_T / \tau)$ — teacher distribution at temperature $\tau$
- $p_S^\tau = \text{softmax}(\text{logits}_S / \tau)$ — student distribution at temperature $\tau$
- $\alpha$: mix between hard labels and soft targets
- $\tau > 1$: softens distributions so small probabilities become more visible

**Temperature $\tau$:** at $\tau=1$, the teacher's distribution is peaked on one class. At $\tau=4$, it's flatter and reveals the teacher's uncertainty — the student can see that "cat" and "lion" are more similar than "cat" and "car."

```
Teacher logits for "cat" image:
  cat: 8.2, tiger: 3.1, dog: 2.8, car: -1.2, pizza: -3.4

Softmax τ=1:  [0.961, 0.020, 0.015, 0.003, 0.001]   ← almost one-hot
Softmax τ=4:  [0.54,  0.17,  0.15,  0.08,  0.06]    ← reveals relationships
```

```python
def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, tau=4.0):
    # Soft targets from teacher
    p_teacher = F.softmax(teacher_logits / tau, dim=-1)
    p_student  = F.log_softmax(student_logits / tau, dim=-1)
    
    kl_loss = F.kl_div(p_student, p_teacher, reduction='batchmean') * tau**2
    ce_loss = F.cross_entropy(student_logits, labels)
    
    return (1 - alpha) * ce_loss + alpha * kl_loss
```

---

## Sequence-Level Distillation for LLMs

For language models, distillation operates on token sequences rather than single-label classification.

### Forward vs Reverse KL

Two natural choices for measuring divergence between teacher $p_T$ and student $p_S$:

**Forward KL:** $D_{\text{KL}}(p_T \| p_S) = \mathbb{E}_{x \sim p_T}[\log p_T(x) - \log p_S(x)]$

- The student is penalized for assigning low probability to things the teacher likes
- **Mean-seeking:** the student spreads probability to cover all teacher modes
- Used in maximum likelihood training (SFT on teacher-generated data)

**Reverse KL:** $D_{\text{KL}}(p_S \| p_T) = \mathbb{E}_{x \sim p_S}[\log p_S(x) - \log p_T(x)]$

- The student is penalized for generating things the teacher assigns low probability to
- **Mode-seeking:** the student picks one or a few modes and focuses on them
- Requires sampling from the student (on-policy)

```
Teacher distribution over responses:
  ████████  (good answer A, prob 0.5)
  █████     (good answer B, prob 0.3)
  ██        (mediocre, prob 0.15)
  █         (bad, prob 0.05)

Forward KL student:        Reverse KL student:
  ████████  (covers A)       ████████████████ (focuses on A only)
  █████     (covers B)       
  ██        (covers mediocre)  
Student matches all modes  Student picks the best mode
```

For generation tasks where we want diverse, high-quality outputs, forward KL (train on teacher samples) works well. For tasks where we want the student to be specifically good at what the teacher is good at, reverse KL is often better.

---

## On-Policy Distillation

The key innovation from the [Thinking Machines blog post](https://thinkingmachines.ai/blog/on-policy-distillation/): **the student generates the trajectories, the teacher scores them token-by-token.**

```
Step 1: Student generates a response
  Prompt: "Solve: 2x + 5 = 11"
  Student rollout: "Let me solve for x. 2x = 11 - 5 = 6, so x = 3."

Step 2: Teacher scores each token
  Teacher log-prob: log p_T("Let" | prompt) = -0.3
                    log p_T("me"  | prompt, "Let") = -0.8
                    log p_T("solve" | ...) = -0.4
                    ...

Step 3: Compute reverse KL as a per-token reward
  r_t = log p_S(a_t | context) - log p_T(a_t | context)
      = how much more probability the student assigns vs the teacher

Step 4: Minimize this with policy gradient (same as PPO/GRPO)
```

The loss per token is the reverse KL divergence conditioned on the student's own trajectory:

$$\mathcal{L}_{\text{on-policy-distill}} = \mathbb{E}_{y \sim p_S}\left[\sum_t \log p_S(y_t | y_{<t}, x) - \log p_T(y_t | y_{<t}, x)\right]$$

The crucial property: **the context $y_{<t}$ is the student's own generation**, not the teacher's. The student learns to match the teacher in its own distributional territory.

```
Off-policy distillation:
  Teacher generates: "Let's use algebra. First isolate x: 2x = 6, x = 3."
  Student trains on THIS sequence → learns teacher's STYLE of reasoning
  But at test time, student generates differently → train/test gap

On-policy distillation:
  Student generates: "I'll try 2x + 5 = 11, so x = 3. Done."
  Teacher scores THIS sequence → student learns to match teacher
                                  on ITS OWN sequences
  No train/test gap → much better generalization
```

### Implementation

The implementation is nearly identical to PPO/GRPO. The only change: replace the reward model (or verifier) with the teacher's log-probabilities.

```python
def on_policy_distillation_step(student, teacher, prompts, beta=0.1):
    # 1. Student generates responses (on-policy rollouts)
    with torch.no_grad():
        outputs = student.generate(prompts, do_sample=True, temperature=1.0)
    
    # 2. Compute log-probs under both models
    student_log_probs = student.log_probs(prompts, outputs)   # (B, T)
    with torch.no_grad():
        teacher_log_probs = teacher.log_probs(prompts, outputs) # (B, T)
    
    # 3. Per-token reverse KL as reward signal
    # r_t = log p_S(y_t) - log p_T(y_t)
    # We want to MINIMIZE reverse KL, so reward = -KL = -(log p_S - log p_T)
    token_rewards = teacher_log_probs - student_log_probs    # (B, T)
    
    # 4. Policy gradient loss (same structure as GRPO/PPO)
    # No need for a value function — use mean reward as baseline
    advantages = (token_rewards - token_rewards.mean()) / (token_rewards.std() + 1e-8)
    
    # REINFORCE-style update
    loss = -(student_log_probs * advantages.detach()).mean()
    
    return loss
```

Note: the teacher only needs a **single forward pass** per response (not a full rollout). Teacher computation is cheap relative to student rollout generation.

---

## Results: Why It Works

The Thinking Machines experiments on AIME'24 (a hard math benchmark):

```
Method                    AIME'24 Score   GPU Hours
──────────────────────────────────────────────────────
SFT baseline (400k prompts)   60%           —
RL (PPO/GRPO)                 67.6%         17,920
On-policy distillation        74.4%         1,800

On-policy distillation: +7% over RL with 10× less compute
```

Why does on-policy distillation converge faster than RL?

- **Denser signal:** RL gives 1 bit per episode (correct/incorrect). Distillation gives information at every token — hundreds of bits per episode.
- **Softer signal:** the teacher's log-probability is a continuous signal, not a binary 0/1. Gradients are smoother and more informative.
- **No reward hacking:** the teacher's token-level signal is harder to game than a sparse verifier.

---

## Application: Recovering from Catastrophic Forgetting

One of the most practical applications: **continual learning**. When you fine-tune a model on a new domain, it often forgets its original capabilities.

```
Baseline Qwen3-8B:
  IF-Eval (instruction following): 85%
  Internal company QA:             18%   ← doesn't know company facts

After mid-training on company documents:
  IF-Eval:      45%   ← catastrophic forgetting!
  Internal QA:  43%   ← learned new knowledge

After on-policy distillation recovery:
  (distill against the pre-finetuning checkpoint)
  IF-Eval:      83%   ← recovered!
  Internal QA:  41%   ← knowledge retained
```

The process:
1. Fine-tune model on new domain data → `model_new` (knows new stuff, forgot old)
2. Keep original checkpoint → `model_original`
3. Run on-policy distillation: `model_new` generates, `model_original` scores
4. `model_new` learns to match `model_original`'s token distributions on its own outputs
5. Result: model that knows new facts AND behaves like the original

This works because the student is still generating in its own distribution (which includes new knowledge). The teacher only pulls the student's *behavior* (instruction following, safety) back toward the original — not its knowledge.

---

## Distillation vs Other Post-Training Methods

```
                SFT         RL           On-Policy Distillation
─────────────────────────────────────────────────────────────────
Data source     Fixed set   Student      Student rollouts
                            rollouts
Supervision     Hard labels Sparse       Dense per-token signal
                (per-token) (per-episode)
Train/test gap  Yes         No           No
Requires        No          Reward model Teacher model
special model              or verifier   (forward pass only)
Convergence     Fast        Slow         Fast
Compute         Low         High         Medium
Best for        Style/format Complex     Closing gap to teacher,
                adaptation  reasoning   continual learning
```

---

## Practical Variants

### Speculative Distillation

Train the student to match the teacher on a **subset** of tokens — specifically the ones where the teacher is most confident. Skip tokens where the teacher is uncertain (they're noisier training signal).

### GKD: Generalized Knowledge Distillation

A unification (Agarwal et al., 2023) that includes on-policy distillation as a special case. The loss is:

$$\mathcal{L}_{\text{GKD}} = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\beta}\left[D_f(p_T(\cdot|x,y_{<t}) \| p_S(\cdot|x,y_{<t}))\right]$$

where $\pi_\beta = \beta\, p_T + (1-\beta)\, p_S$ interpolates between sampling from the teacher ($\beta=1$, off-policy) and the student ($\beta=0$, on-policy). Setting $\beta=0$ recovers on-policy distillation.

### Token-Level DPO Distillation

Some approaches combine DPO-style preference optimization with distillation: treat teacher generations as "chosen" and student generations as "rejected," then run DPO. This avoids policy gradient entirely and uses a simpler supervised loss.

---

## Summary

```
Classic distillation:    student learns from teacher's soft labels
                         on FIXED data → off-policy, fast but limited

On-policy distillation:  student generates its own data
                         teacher scores each token → on-policy + dense signal
                         Best of both worlds: no train/test gap + rich supervision

Key use cases:
  1. Compress large teacher → smaller student (speed/cost)
  2. Recover from catastrophic forgetting after fine-tuning
  3. Transfer a capability discovered by RL without running full RL
  4. Distill reasoning traces from o1/R1 into smaller models

The RL interpretation:
  On-policy distillation = RL where the reward is "be more like the teacher"
  This is softer and denser than task reward → faster convergence
```

See also: [posttraining.md](posttraining.md) for RLHF and DPO, and [reasoning_models.md](reasoning_models.md) for how DeepSeek-R1 uses distillation to transfer reasoning from large RL-trained models into smaller ones.
