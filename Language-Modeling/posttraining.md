# Post-Training: SFT, RLHF, DPO, and Beyond

Pretraining gives a model that predicts text. Post-training makes it a useful assistant: helpful, harmless, honest. The key insight (Ouyang et al., InstructGPT, 2022) is that **human preferences are the training signal**.

---

## Supervised Fine-Tuning (SFT)

The first post-training step: fine-tune the model on high-quality demonstrations of desired behavior.

**Data format:** instruction-response pairs:
```
System: You are a helpful assistant.
User: Explain gradient descent in simple terms.
Assistant: Gradient descent is like rolling a ball downhill...
```

**Objective:** same CLM loss, but only on the assistant's response tokens (not the user's prompt):

$$\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{response}} \log P_\theta(x_t \mid x_{<t})$$

**Data quality > quantity.** InstructGPT used ~13k demonstrations; quality hand-curated by contractors vastly outweighs 100k noisy examples. This has become a consensus view.

### SFT data sources

- **Human-written:** expensive, gold standard (OpenAI's contractor data, ShareGPT)
- **Distilled from stronger models:** LLaMA used GPT-4 outputs (controversial legally)
- **Synthetic (self-instruct):** prompt the model itself to generate Q&A pairs, filter, use as data; enables iterative self-improvement
- **Template-generated:** structured data from knowledge bases → question/answer pairs

### SFT limitations

SFT teaches the model to **imitate** demonstrations. But:
- Demonstrations can't capture all the nuances of human preference
- The model learns to produce responses that look like the demonstrations, not necessarily the best responses
- Imitation doesn't generalize well to novel situations

This motivates reinforcement learning from human feedback.

---

## RLHF: Reinforcement Learning from Human Feedback

The full InstructGPT/ChatGPT pipeline adds two more steps after SFT.

### Step 1: Train a Reward Model

Collect human **comparisons** (A vs B), not demonstrations. Human labelers rank two model responses to the same prompt.

The reward model $r_\phi(x, y)$ scores a response $y$ given prompt $x$. Train with the Bradley-Terry preference model:

$$P(y_w \succ y_l) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

Loss:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

The reward model is typically initialized from the SFT model with a linear head replacing the LM head.

**Why comparisons and not absolute scores?** Humans are inconsistent at absolute ratings ("7/10") but reliable at relative comparisons ("A is better than B"). The Bradley-Terry model is designed for pairwise comparisons.

### Step 2: RLHF with PPO

Fine-tune the SFT model using the reward model as a reward function. Use PPO (Proximal Policy Optimization) as the RL algorithm.

The objective:

$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}\left[r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right]$$

- First term: maximize reward
- KL penalty: don't deviate too far from the SFT policy $\pi_{\text{ref}}$

The KL penalty prevents **reward hacking**: the model finding degenerate responses that trick the reward model (e.g., very long outputs, repeated text).

**PPO update:**

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\!\left(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

where $r_t(\theta) = \pi_\theta / \pi_{\text{old}}$ is the importance ratio and $A_t$ is the advantage estimate.

**PPO is complex.** It requires:
1. The policy model ($\pi_\theta$, the LLM being trained)
2. The reference model ($\pi_{\text{ref}}$, SFT model, frozen)
3. The reward model ($r_\phi$, frozen)
4. The value model (critic, for advantage estimation)

All four models must be loaded simultaneously → 4× the memory of the base model. This is why PPO is expensive.

---

## DPO: Direct Preference Optimization

DPO (Rafailov et al., 2023) eliminates the need for the reward model and PPO by showing that the RLHF objective has a **closed-form optimal policy**:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)$$

Rearranging, the reward can be expressed in terms of the policy:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

Substituting into the Bradley-Terry preference model and using the fact that $Z(x)$ cancels in pairwise comparisons:

$$P(y_w \succ y_l) = \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

**DPO loss:**

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \underbrace{\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}}_{\text{implicit reward (winner)}} - \beta \underbrace{\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}}_{\text{implicit reward (loser)}}\right)\right]$$

**What DPO does:** increase the probability of winning responses relative to the reference model, decrease the probability of losing responses — with the magnitude governed by how confident the reference model was.

```python
def dpo_loss(policy, ref_policy, prompt, chosen, rejected, beta=0.1):
    # Log probs under policy
    log_p_chosen = policy.log_prob(chosen, given=prompt)
    log_p_rejected = policy.log_prob(rejected, given=prompt)
    # Log probs under reference
    log_ref_chosen = ref_policy.log_prob(chosen, given=prompt)
    log_ref_rejected = ref_policy.log_prob(rejected, given=prompt)
    
    chosen_reward = beta * (log_p_chosen - log_ref_chosen)
    rejected_reward = beta * (log_p_rejected - log_ref_rejected)
    
    loss = -F.logsigmoid(chosen_reward - rejected_reward)
    return loss.mean()
```

**Advantages of DPO over PPO:**
- No reward model needed
- No RL training loop, value model, or on-policy sampling
- Stable training — just supervised learning on preference data
- 2–4× less compute

**Disadvantages:**
- Uses offline preference data (not on-policy). The winning/losing responses were generated by a different model → distributional mismatch
- Empirically, PPO can outperform DPO on very challenging tasks (RLVR, see below)
- Mode-seeking: DPO can collapse the policy in subtle ways

---

## DPO variants

**IPO (Identity Preference Optimization):** replaces the log-sigmoid with a squared error, avoiding over-confidence issues in DPO.

**KTO:** uses binary feedback (good/bad) instead of pairwise comparisons, using a prospect-theory-inspired objective. More data-efficient.

**SimPO:** removes the reference model by using length-normalized log-probs. Simpler, no ref model needed.

**ORPO:** combines SFT loss and preference loss into one, training from scratch without a separate SFT phase.

---

## Constitutional AI (CAI) and RLAIF

**RLAIF (RL from AI Feedback):** replace human annotators with another LLM for preference labeling. A judge model scores responses, generating synthetic preference data at scale.

**Constitutional AI (Anthropic):** define a set of principles ("the constitution"). Use an AI to:
1. Critique model responses against the constitution
2. Revise them to be compliant
3. Use the critiques + revisions to train a reward model or directly fine-tune

Enables scalable alignment without human labelers for every comparison.

---

## Reward hacking and overoptimization

As we optimize harder against the reward model, the policy eventually finds responses that score highly on $r_\phi$ but are actually bad. The reward model is an imperfect proxy.

**Overoptimization curve:** reward model score increases, then actual human preference peaks and declines as the policy diverges from natural text.

The KL penalty in RLHF is specifically designed to slow this. But it's not foolproof: with enough optimization, the model learns to exploit reward model blind spots.

**Mitigation:** iterative RLHF (collect new human comparisons on the updated model, retrain the reward model, repeat).
