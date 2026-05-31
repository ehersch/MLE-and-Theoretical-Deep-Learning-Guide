# RL for Large Language Models

Applying RL to language models requires rethinking the MDP formulation. The "environment" is text generation, the "agent" is the LLM, and the reward signal comes from human preference, task success, or verifiable correctness. This section goes deep on the RL mechanics specific to language.

---

## The Token-Level MDP

Language generation is a sequential decision process. Formalize it as an MDP:

```
State s_t:     the sequence of tokens generated so far
               s_t = (x_1, x_2, ..., x_t)   [prompt + generated tokens]

Action a_t:    the next token to generate
               a_t ∈ V   [vocabulary, |V| ~ 32k-128k]

Transition:    deterministic: s_{t+1} = (s_t, a_t)
               (append the new token)

Reward r_t:    mostly 0 during generation
               r_T = R(prompt, full_response)  at end of sequence

Policy:        π_θ(a_t | s_t) = LLM's next-token distribution
               = softmax(LM_head(transformer(s_t)))
```

**Key properties:**
- **Discrete action space:** vocabulary ≈ 100k tokens → can't use continuous-action methods (SAC)
- **Very long episodes:** responses can be 1000+ tokens → severe credit assignment
- **Terminal reward:** reward comes at the end → sparse, delayed signal
- **Huge state space:** $|V|^T$ possible sequences

---

## Why PPO for Language (Not Q-Learning)?

**Q-learning requires $\arg\max_a Q(s,a)$** over vocabulary. With $|V| = 100{,}000$ tokens, this is expensive. Computing $Q(s, a)$ for all $a$ at each step is a full forward pass × $|V|$ — impractical.

**Policy gradient methods** compute $\nabla_\theta \log \pi_\theta(a|s)$ directly from the sampled action — $O(1)$ in action space. Perfect for discrete, large action spaces.

---

## RLHF with PPO for LLMs

The InstructGPT / ChatGPT pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RLHF Training Setup                         │
│                                                                 │
│  4 models loaded simultaneously:                                │
│                                                                 │
│  1. Policy π_θ        (LLM being trained, ~7B params)          │
│  2. Reference π_ref   (SFT model, frozen)                       │
│  3. Reward model R_φ  (trained on human preferences, frozen)    │
│  4. Critic V_ψ        (value function, separate network)        │
│                                                                 │
│  Memory: 4 × model_size + activations → 4-8 A100s minimum      │
└─────────────────────────────────────────────────────────────────┘
```

**Objective:**

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}\!\left[R_\phi(x, y) - \beta \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right]$$

The KL term:

$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

can be computed per-token (since both models see the same sequence):

$$\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} = \sum_{t=1}^T \log\frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{\text{ref}}(y_t|x, y_{<t})}$$

**Per-token reward:** the terminal reward $R_\phi(x,y)$ is assigned to the **last token**. The per-token KL penalty is subtracted at every step:

$$r_t = \begin{cases} R_\phi(x,y) - \beta \log\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\text{ref}}(y_t|x,y_{<t})} & t = T \\ -\beta \log\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\text{ref}}(y_t|x,y_{<t})} & t < T \end{cases}$$

This turns the language generation into a proper per-step MDP for PPO.

**PPO for LLMs — training loop:**

```python
def ppo_language_step(policy, ref_policy, reward_model, critic, prompts):
    # 1. Generate responses with current policy
    responses = policy.generate(prompts, temperature=1.0)
    
    # 2. Score with reward model
    R = reward_model(prompts, responses)              # (B,) scalar per response
    
    # 3. Compute per-token KL penalties
    log_probs_policy = policy.log_probs(prompts, responses)    # (B, T)
    log_probs_ref    = ref_policy.log_probs(prompts, responses) # (B, T)
    kl_per_token = log_probs_policy - log_probs_ref             # (B, T)
    
    # 4. Assemble per-token rewards
    rewards = -beta * kl_per_token                             # (B, T)
    rewards[:, -1] += R                                         # add terminal reward
    
    # 5. Compute values and advantages (GAE)
    values = critic(prompts, responses)                          # (B, T)
    advantages = compute_gae(rewards, values)
    
    # 6. PPO update (multiple epochs)
    for _ in range(n_ppo_epochs):
        log_probs_new = policy.log_probs(prompts, responses)
        ratio = (log_probs_new - log_probs_policy.detach()).exp()  # (B, T)
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1-eps, 1+eps) * advantages
        actor_loss  = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(critic(prompts, responses), (advantages + values).detach())
        (actor_loss + 0.5 * critic_loss).backward()
```

---

## GRPO: Group Relative Policy Optimization

GRPO (DeepSeek, 2024) — a simpler, more scalable alternative to PPO for language. Used for DeepSeek-R1.

**Key simplification:** eliminate the critic network. Instead, estimate the baseline from a **group of samples** for the same prompt.

```
For each prompt x:
    Generate G responses: y_1, y_2, ..., y_G (e.g., G=8)
    Score each: r_1, r_2, ..., r_G via reward model or verifier
    
    Baseline = mean(r_1, ..., r_G)
    Advantage of y_i = r_i - mean({r_j})
    
    Optionally normalize: Â_i = (r_i - mean) / std
```

GRPO loss:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^G \min\!\left(\frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)} \hat{A}_i,\ \text{clip}\!\left(\frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}, 1\pm\varepsilon\right)\hat{A}_i\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

```python
def grpo_loss(policy, ref_policy, prompts, G=8, beta=0.01, eps=0.2):
    losses = []
    for prompt in prompts:
        # Sample G responses
        responses = [policy.generate(prompt) for _ in range(G)]
        rewards = torch.tensor([reward_fn(prompt, r) for r in responses])
        
        # Normalize advantages within group
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        for response, adv in zip(responses, advantages):
            log_p = policy.sequence_log_prob(prompt, response)
            log_p_old = log_p.detach()
            log_p_ref = ref_policy.sequence_log_prob(prompt, response)
            
            ratio = (log_p - log_p_old).exp()
            surr = torch.min(ratio * adv, ratio.clamp(1-eps, 1+eps) * adv)
            kl = log_p - log_p_ref
            losses.append(-surr + beta * kl)
    
    return torch.stack(losses).mean()
```

**Why GRPO > PPO for LLMs:**

| | PPO | GRPO |
|---|---|---|
| Value network | Required (4th model) | Not needed |
| Memory | 4× model | 2× model (policy + ref) |
| Baseline | Critic $V(s)$ | Sample mean across group |
| Variance | Lower (learned critic) | Higher (sample estimate) |
| Simplicity | Complex | Simple |

For **verifiable rewards** (math, code) where the reward signal is strong and binary, GRPO works excellently — the high variance of the group baseline doesn't matter when you have many samples and a clear signal.

---

## RLVR: RL with Verifiable Rewards

The key insight behind DeepSeek-R1 and o1-class models.

**For math and code:** reward can be verified automatically:

$$r(y) = \begin{cases} +1 & \text{if answer is correct (verified by parser/unit tests)} \\ 0 & \text{otherwise} \end{cases}$$

No reward model needed! This eliminates:
- Reward model training cost
- Reward model distribution shift / hacking
- Human annotation for comparisons

**The training signal is perfectly aligned with task performance.** PPO/GRPO with a binary verifier is extremely effective.

```
RLVR loop:
  Sample math problem x
  Generate G responses {y_i} with chain-of-thought
  Check each final answer against ground truth → {r_i ∈ {0,1}}
  Use GRPO to update policy:
    Responses that got the right answer: positive advantage
    Responses that got the wrong answer: negative advantage
  
  After enough training:
    Model spontaneously develops longer reasoning chains
    Self-correction ("wait, that's wrong...")
    Multi-step verification
```

**Why this works for reasoning:** correctness is all-or-nothing. The model can't partially game this reward. It must actually solve the problem to get the signal.

---

## Format Rewards and Length Penalties

Beyond correctness, training often includes auxiliary rewards:

```python
def composite_reward(prompt, response, answer):
    # Core reward: correctness
    r_correct = float(extract_answer(response) == answer)
    
    # Format reward: did the model follow the format?
    r_format = float("<answer>" in response and "</answer>" in response)
    
    # Length penalty: discourage verbosity (logarithmic)
    r_length = -0.001 * max(0, len(response) - target_length)
    
    # Repetition penalty
    r_repeat = -count_repeated_phrases(response) * 0.01
    
    return r_correct + 0.1 * r_format + r_length + r_repeat
```

**KL coefficient schedule:** often anneal $\beta$ (KL penalty) during training:
- High $\beta$ early → stay close to reference, stable
- Low $\beta$ late → allow more exploration, better final performance

---

## Comparison: PPO vs DPO vs GRPO

```
                   PPO           DPO          GRPO
Signal type       RM score     Preference    Verifiable
Training          Online        Offline       Online
Models needed     4             2             2
Data              Generated     Preference    Generated
                               pairs
Best for          General       General       Math/code
                  alignment     alignment     (RLVR)
Reward hacking    Medium risk   Low risk      Very low
                  (RM imperfect)(no RM)       (verifier)
Complexity        High          Low           Medium
```

---

## The Full LLM Post-Training Picture

```
Pretrained model
     ↓
SFT (imitation learning on demonstrations)
     ↓
RLHF Phase 1: Reward model training on human comparisons
     ↓
RLHF Phase 2: PPO/GRPO optimization against reward model
     ↓
Optional: DPO refinement on preference data
     ↓
Optional: RLVR on verifiable tasks (math, code)
     ↓
Production model
```

See also: [posttraining.md](../Language-Modeling/posttraining.md) for the machine learning perspective, and [reasoning_models.md](../Language-Modeling/reasoning_models.md) for RLVR and o1-style training in depth.
