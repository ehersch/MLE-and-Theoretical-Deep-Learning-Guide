# Meta-Reinforcement Learning

Standard RL trains a policy for one task. **Meta-RL** trains an agent that can **rapidly adapt** to new tasks — learning to learn, not just learning.

---

## The Motivation: Slow Human vs Fast Human

```
Standard RL agent learning to walk:
  Millions of trials. No prior knowledge.

Human child learning to walk:
  Hundreds of trials. Has prior knowledge about physics,
  body structure, balance from watching others.

Meta-RL agent (ideal):
  First 5 trials: explore task specifics
  Next 5 trials: near-optimal behavior
  ← leverages prior experience across many tasks
```

Meta-RL is sometimes called "few-shot RL" — the agent should achieve good performance with very few trials on a new task.

---

## Problem Setup

A distribution of tasks $p(\mathcal{T})$. At **meta-train** time, the agent sees many tasks from this distribution. At **meta-test** time, it's given a new task (also from $p(\mathcal{T})$) and must adapt quickly.

```
Meta-train:                  Meta-test:
  Task 1: reach goal A  │      New task: reach goal Z (unseen)
  Task 2: reach goal B  │        Episode 1: explore
  Task 3: reach goal C  │        Episode 2: exploit prior experience
  ...                   │        → reaches goal Z quickly!
```

---

## RL² : Learning a Learning Algorithm

**RL²** (Duan et al., 2016; Wang et al., 2016): the meta-learner is an **RNN**. The entire "learning algorithm" is implemented by the RNN's hidden state.

**Key idea:** the RNN's hidden state $h_t$ accumulates experience across the episode. The RNN learns to use this memory to adapt its own behavior — it literally learns a learning algorithm in its weights.

```
Architecture:
  Input at step t: (s_t, a_{t-1}, r_{t-1}, done_{t-1})
  Hidden state: h_t  ← accumulates task-relevant information
  Output: π(a_t | h_t)  ← policy conditioned on accumulated experience

One trial across multiple episodes (same task):
  s_0 → h_1 → a_0 → r_0 → s_1 → h_2 → a_1 → r_1 → ... → s_T (end of trial)
  s_0 → h_1' → a_0' → r_0' → ... (2nd episode: hidden state carries over!)
```

**Training:** standard RL (PPO/TRPO) across the meta-training distribution. Each rollout spans multiple episodes of the same task, with hidden state carried across episode boundaries.

**What the RNN learns:** on episode 1, explore (uncertainty in hidden state → exploration behavior). On episode 2, exploit (hidden state has identified the task → optimal policy).

```python
# RL² forward pass sketch
class RL2_Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Inputs: obs + prev_action + prev_reward + done_flag
        self.gru = nn.GRU(obs_dim + action_dim + 2, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs, prev_action, prev_reward, done, h):
        # Concatenate all inputs
        x = torch.cat([obs, prev_action, prev_reward.unsqueeze(-1), 
                        done.unsqueeze(-1).float()], dim=-1)
        out, h_new = self.gru(x.unsqueeze(1), h)
        return self.actor(out.squeeze(1)), self.critic(out.squeeze(1)), h_new
```

**RL² vs standard RL:**
```
Standard RL:  policy trained from scratch for each task
RL²:          RNN trained once; adapts in-context at test time
              (like in-context learning in LLMs!)
```

---

## MAML: Model-Agnostic Meta-Learning (for RL)

**MAML** (Finn et al., 2017): instead of learning an adaptive RNN, learn **initial parameters** $\theta$ such that a few gradient steps lead to good performance on any task.

```
Meta-objective:
  Find θ such that θ + k·gradient_steps_on_task_T
  = good policy for task T, for any T ~ p(T)

Meta-update:
  θ ← θ - α · ∇_θ [ Σ_T L_T(θ - β∇_θ L_T(θ)) ]
              ↑                    ↑
           outer loop          inner loop
         (meta-learning)    (task adaptation)
```

**Intuition:** MAML finds a parameter initialization that is **close to all tasks** in the loss landscape — a point from which a few gradient steps can reach any task's optimum.

```
Loss landscape over tasks:
        T1 optimal              T2 optimal
            ×                      ×
             \                    /
              \                  /
               ×  MAML θ_0  ×
              / \            / \
             /   \          /   \
            ×     ×        ×     ×
         Task 1         Task 2
         optimum        optimum
         (after 1 step) (after 1 step)
```

**Algorithm:**
```
For each meta-update:
    Sample batch of tasks {T_i}
    For each T_i:
        Collect a few trajectories under θ
        Compute task gradient: g_i = ∇_θ L_{T_i}(θ)
        Compute adapted params: θ_i' = θ - β·g_i
        Collect trajectories under θ_i' → compute L_{T_i}(θ_i')
    Meta-update: θ ← θ - α·∇_θ [ Σ_i L_{T_i}(θ_i') ]
```

The meta-gradient requires differentiating through the inner-loop gradient step — a second-order operation. Computationally expensive.

**MAML-RL results:** 2 gradient steps on a new MuJoCo locomotion task → near-optimal performance. Without MAML: thousands of steps.

---

## Bayesian RL and Uncertainty

The meta-RL problem can be framed as: the agent has uncertainty over which task it's in, and should maintain a belief over tasks, using Bayes' rule.

$$P(\mathcal{T} | \tau_{1:t}) \propto P(\tau_{1:t} | \mathcal{T}) P(\mathcal{T})$$

The optimal meta-policy is then:

$$\pi^*(a | s, \tau_{1:t}) = \int \pi^*_{\mathcal{T}}(a|s) P(\mathcal{T}|\tau_{1:t}) d\mathcal{T}$$

RL² implicitly implements this belief update — the RNN hidden state is an amortized approximate posterior over tasks.

---

## Meta-RL ↔ In-Context Learning in LLMs

There's a deep connection between meta-RL and in-context learning (ICL) in large language models.

```
Meta-RL:
  Training: many tasks → RNN learns to adapt in-context
  Test time: few trials → RNN's hidden state = task belief

LLM in-context learning:
  Pretraining: many text tasks (implicit) → transformer learns to adapt
  Test time: few-shot examples → attention = task belief

Both:
  The "weights" are fixed at test time.
  Adaptation happens through in-context information.
  No gradient updates at test time.
```

A sufficiently large language model pretrained on diverse text has implicitly done meta-learning — it can do few-shot RL-like tasks in context (e.g., "here are some examples of correct solutions, now solve this new problem").

This connection motivates using pretrained LLMs as the backbone for meta-RL agents (see [rl_for_llms.md](rl_for_llms.md)).
