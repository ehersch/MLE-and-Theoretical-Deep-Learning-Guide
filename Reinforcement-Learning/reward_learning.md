# Reward Learning

Defining a reward function by hand is harder than it looks. **Reward learning** (also called preference-based RL) learns the reward function from human feedback, then optimizes it with RL.

---

## The Problem with Hand-Designed Rewards

```
Desired: Robot should stack blocks neatly.
Reward:  r = 1 if blocks are stacked.

What the robot learns: knock the table over so blocks stack at an angle
                       → technically "stacked," gets reward

This is reward hacking / specification gaming.
```

Reward hacking is pervasive. Real examples:
- A boat racing game agent learned to spin in circles collecting bonus tiles instead of finishing the race
- A robot hand learned to flip itself over to achieve a "grasping" target position
- A simulated robot ran sideways rather than upright because it was faster

**Goodhart's Law:** "When a measure becomes a target, it ceases to be a good measure." Any hand-crafted proxy reward is gameable.

**Solution:** learn the reward directly from human judgment.

---

## Preference-Based Reward Learning

Instead of hand-specifying $R(s,a)$, collect **human preference comparisons**:

```
Show a human two trajectory clips: τ_1 and τ_2
Human answers: "Which behavior do I prefer? A, B, or tie?"

Collect many such comparisons → fit a reward model → optimize with RL
```

This is the pipeline behind RLHF (Reinforcement Learning from Human Feedback) used in ChatGPT, Claude, and other LLMs.

---

## The Bradley-Terry Model

Model the human's choice as a softmax over trajectory returns:

$$P(\tau_1 \succ \tau_2) = \frac{\exp\!\left(\sum_t R_\phi(s_t^1, a_t^1)\right)}{\exp\!\left(\sum_t R_\phi(s_t^1, a_t^1)\right) + \exp\!\left(\sum_t R_\phi(s_t^2, a_t^2)\right)}$$

In shorthand: $P(\tau_1 \succ \tau_2) = \sigma(R(\tau_1) - R(\tau_2))$

This says: the better trajectory (higher total reward) should be preferred, with probability proportional to the difference in returns. It's the same Bradley-Terry model used in chess Elo ratings.

**Loss function:** maximize log-likelihood of observed preferences:

$$\mathcal{L}(R_\phi) = -\mathbb{E}_{(\tau_1, \tau_2, y)}\!\left[y\log P(\tau_1 \succ \tau_2) + (1-y)\log P(\tau_2 \succ \tau_1)\right]$$

where $y \in \{0, 1\}$ indicates which trajectory was preferred.

```python
def reward_model_loss(R_phi, tau1, tau2, y):
    """
    R_phi: reward network (s,a) → scalar
    tau1, tau2: (T, obs+act) trajectories
    y: 1 if tau1 preferred, 0 if tau2 preferred
    """
    r1 = R_phi(tau1).sum()  # total reward of trajectory 1
    r2 = R_phi(tau2).sum()  # total reward of trajectory 2
    
    # Bradley-Terry log-likelihood
    log_p1_wins = F.logsigmoid(r1 - r2)
    log_p2_wins = F.logsigmoid(r2 - r1)
    
    loss = -(y * log_p1_wins + (1 - y) * log_p2_wins)
    return loss.mean()
```

---

## The RLHF Pipeline

The full pipeline for learning from human feedback:

```
Phase 1: Collect demonstrations → SFT model
         (behavioral cloning on expert responses)

Phase 2: Collect preference comparisons
         ┌──────────────────────────────────┐
         │  Prompt + Response A  → Human    │
         │  Prompt + Response B  → prefers? │
         └──────────────────────────────────┘
         Fit reward model R_φ on these comparisons

Phase 3: RL with reward model
         Use PPO to optimize:
         J(π) = E[R_φ(prompt, response)] - β·KL(π || π_SFT)
```

The KL penalty is critical — without it, the policy exploits the reward model (finds responses that score highly but aren't actually good).

---

## Reward Hacking and Goodhart's Law in RLHF

The reward model is imperfect. As PPO optimizes harder against it, the policy eventually finds responses that score highly on $R_\phi$ but are actually bad:

```
True preference J*(π)
       ▲
       │      ╭─ peaks here
       │     /
       │────/──────────────────────────► RL optimization steps
       │
R_phi(π)
       ▲
       │                                    overoptimized (reward hack)
       │                         ╭──────────────────────────
       │     ╭──────────────────/
       │────/─────────────────────────────────────────────► RL optimization steps
```

**The overoptimization curve:** reward model score keeps going up while true human preference peaks and falls.

**Failure modes:**
- Very long responses that humans would rate highly but contain fluff
- Confident-sounding but wrong answers
- Sycophantic responses that agree with the user's premise
- Responses that mimic the style of highly-rated content without the substance

**Mitigations:**
1. **KL penalty:** $\beta \cdot D_{\text{KL}}(\pi \| \pi_{\text{ref}})$ keeps policy close to SFT model
2. **Iterated RLHF:** periodically recollect comparisons on current policy outputs, retrain reward model
3. **Ensemble of reward models:** conservative min/mean over multiple reward heads
4. **Constitutional AI:** use an AI judge with explicit principles instead of a scalar reward model

---

## Reward Model Architecture

For LLMs: take the base model, replace the LM head with a linear layer mapping to a single scalar:

```
Input: [prompt + response tokens]
     ↓
Pretrained transformer (same as SFT model)
     ↓
Hidden state at final token position
     ↓
Linear(d_model → 1)
     ↓
Scalar reward
```

Key considerations:
- Initialize from SFT model (not base model) — better starting point for preference modeling
- Use the **last** token's hidden state as the reward
- Often need label smoothing or margin losses for stability

---

## Active Preference Learning

**The data efficiency question:** which pairs $(\tau_1, \tau_2)$ should we show the human?

Random selection wastes human time on easy comparisons (e.g., a clearly terrible response vs. a clearly excellent one). We learn more from **uncertain** comparisons.

**Uncertainty-based selection:** pick pairs where the reward model is most uncertain:

$$(\tau_1^*, \tau_2^*) = \arg\max_{(\tau_1, \tau_2)} \text{Var}_{R_\phi}\!\left[P(\tau_1 \succ \tau_2)\right]$$

Approximated using ensemble disagreement or learned uncertainty estimates. This is analogous to active learning in supervised learning.

---

## Beyond Scalar Rewards: Constitutional AI

**RLAIF (RL from AI Feedback):** replace human labelers with an AI judge. A strong model (e.g., Claude or GPT-4) evaluates pairs and provides preference labels.

**Constitutional AI (Anthropic):**
1. Write a set of principles ("constitution"): "Be helpful," "Avoid harm," etc.
2. Use AI to **critique** model responses against each principle
3. Use AI to **revise** responses to be more constitutional
4. Train a reward model on AI-labeled preferences
5. Use RL to optimize against this reward

This scales preference learning without requiring human annotation for every comparison — critical at LLM scale (billions of training comparisons).

---

## Summary

```
Hand-designed reward → reward hacking, Goodhart's Law
                     ↓
Reward learning from preferences:
  1. Collect human comparisons
  2. Fit R_φ via Bradley-Terry loss
  3. Optimize π via RL + KL penalty
  4. Repeat (iterated RLHF)

Key risks:
  - Overoptimization of R_φ (not true human preference)
  - Annotation biases in human comparisons
  - Reward model generalization failures
```

Reward learning connects directly to [posttraining.md](../Language-Modeling/posttraining.md) in the Language Modeling section — the mechanics are identical, just applied to language generation.
