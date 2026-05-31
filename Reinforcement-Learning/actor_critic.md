# Actor-Critic Methods

Actor-critic methods combine the best of value-based and policy gradient methods: the **actor** is the policy $\pi_\theta$, the **critic** is the value function $V_\phi$ or $Q_\phi$ that evaluates the actor's choices.

```
             ┌───────────────────────────────────┐
             │           Environment              │
             └────────────┬──────────────────────┘
                    s_t, r_t │
             ┌──────────────▼──────────────────┐
             │           ACTOR                  │
             │       π_θ(a | s)                 │─── a_t ──►
             │   (learns WHAT to do)            │
             └──────────────────────────────────┘
                           │ s_t
             ┌─────────────▼───────────────────┐
             │           CRITIC                 │
             │   V_φ(s) or Q_φ(s,a)            │
             │  (tells actor HOW WELL it did)   │
             └──────────────────────────────────┘
                    │ advantage Â_t
                    ▼
             Actor updates: ∇_θ log π_θ(a_t|s_t) · Â_t
```

The critic reduces variance (compared to pure REINFORCE) by providing a learned baseline.

---

## Advantage Actor-Critic (A2C)

Two networks share a backbone, with two output heads:

```python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),     nn.ReLU()
        )
        self.actor_head  = nn.Linear(128, n_actions)  # policy logits
        self.critic_head = nn.Linear(128, 1)           # V(s)
    
    def forward(self, x):
        h = self.shared(x)
        dist = torch.distributions.Categorical(logits=self.actor_head(h))
        value = self.critic_head(h).squeeze(-1)
        return dist, value

def a2c_update(model, optimizer, states, actions, returns, gamma=0.99):
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    returns = torch.FloatTensor(returns)
    
    dist, values = model(states)
    advantages = returns - values.detach()
    
    actor_loss  = -(dist.log_prob(actions) * advantages).mean()
    critic_loss = nn.MSELoss()(values, returns)
    entropy_bonus = -dist.entropy().mean()  # encourage exploration
    
    loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_bonus
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**Entropy bonus:** add $-\beta H(\pi)$ to the loss to discourage premature convergence to deterministic policies. The policy is "rewarded" for being uncertain — useful exploration regularizer.

---

## Trust Region Policy Optimization (TRPO)

**The problem with naive policy gradient:** if we take too large a step, the new policy can be drastically worse. There's no guarantee that gradient ascent in parameter space corresponds to improvement in policy space.

```
Policy space J(θ):
   ─────────────────────────
    current policy → small step → still reasonable
    current policy → large step → could land anywhere!
```

**TRPO** (Schulman et al., 2015): constrain each update to stay within a **trust region** in policy space (measured by KL divergence):

$$\max_\theta \; \mathcal{L}_{\text{surrogate}}(\theta) \quad \text{s.t.} \quad \mathbb{E}_s\!\left[D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))\right] \leq \delta$$

The **surrogate objective** $\mathcal{L}$ approximates $J(\theta)$ near $\theta_{\text{old}}$ using importance sampling:

$$\mathcal{L}(\theta) = \mathbb{E}_t\!\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} A_t\right]$$

**Why importance sampling?** We collected data with $\pi_{\theta_{\text{old}}}$, but want to evaluate $\pi_\theta$. IS reweights the old samples to estimate what would have happened under the new policy.

**Solving the constrained problem:** TRPO uses the **conjugate gradient method** to compute the natural gradient and a line search to enforce the constraint. This is complex and computationally expensive — second-order optimization.

---

## PPO: Proximal Policy Optimization

PPO (Schulman et al., 2017) keeps TRPO's trust-region intuition but is far simpler: instead of a hard KL constraint, **clip the importance ratio** directly in the objective.

### The PPO-Clip Objective

Define the probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

PPO clips this ratio to $[1-\varepsilon, 1+\varepsilon]$ (typically $\varepsilon = 0.2$):

$$\mathcal{L}_{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta) A_t, \;\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_t\right)\right]$$

**Intuition via cases:**

```
Case 1: A_t > 0  (action was good — we want to do more of it)
    r_t > 1+ε: clipped → don't push policy too far toward this action
    r_t ≤ 1+ε: unclipped → reward taking this action more

Case 2: A_t < 0  (action was bad — we want to do less of it)
    r_t < 1-ε: clipped → don't push policy too far away from this action
    r_t ≥ 1-ε: unclipped → penalize taking this action

┌──────────────────────────────────────────────────────────┐
│ A_t > 0: want r_t > 1                                    │
│                                                          │
│  L_CLIP                                                  │
│  ▲                                                       │
│  │          ┌──────── clipped (no extra benefit)         │
│  │         /                                             │
│  │        /                                             │
│  │───────/──────────────────────────► r_t               │
│  │    1-ε    1    1+ε                                    │
│                                                          │
│ A_t < 0: want r_t < 1                                    │
│  L_CLIP                                                  │
│  ▲                                                       │
│  │                \                                      │
│  │                 \                                     │
│  │──────────────────\────────────────► r_t               │
│  │    1-ε    1    1+ε                                    │
│          clipped (no extra penalty)                      │
└──────────────────────────────────────────────────────────┘
```

The `min` ensures we never take an **overly optimistic** step: if the ratio would already make things better, we don't push further.

### Full PPO Implementation

```python
import torch, torch.nn as nn
import numpy as np

class PPO:
    def __init__(self, obs_dim, n_actions, lr=3e-4, gamma=0.99,
                 lam=0.95, eps_clip=0.2, n_epochs=10):
        self.model = ActorCritic(obs_dim, n_actions)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma, self.lam, self.eps_clip = gamma, lam, eps_clip
        self.n_epochs = n_epochs
    
    def collect_rollout(self, env, n_steps=2048):
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        s = env.reset()
        for _ in range(n_steps):
            s_t = torch.FloatTensor(s)
            dist, val = self.model(s_t)
            a = dist.sample()
            s_next, r, done, _ = env.step(a.item())
            states.append(s); actions.append(a); rewards.append(r)
            dones.append(done); log_probs.append(dist.log_prob(a))
            values.append(val); s = s_next
        return states, actions, rewards, dones, log_probs, values
    
    def update(self, states, actions, rewards, dones, old_log_probs, old_values):
        # Compute GAE advantages
        advantages = compute_gae(rewards, old_values + [0], dones,
                                 self.gamma, self.lam)
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + torch.tensor(old_values)
        
        states = torch.FloatTensor(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs).detach()
        
        for _ in range(self.n_epochs):
            dist, values = self.model(states)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # PPO clip loss
            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.opt.zero_grad(); loss.backward(); 
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.opt.step()
```

### Why PPO Dominates

- Simple to implement (no second-order optimization like TRPO)
- Works on-policy but can do multiple gradient steps per rollout (`n_epochs`)
- The clipping provides implicit trust region
- State-of-the-art on many continuous and discrete control tasks
- **This is the same algorithm used for RLHF (with modifications)**

---

## TRPO vs PPO

```
TRPO:                            PPO:
  Hard KL constraint               Soft clip on ratio
  Second-order optimization        First-order (Adam)
  Conjugate gradient + line search Simple gradient step × K
  Theoretically principled         Empirically superior
  Hard to implement                ~50 lines of code
```

PPO is the workhorse of modern deep RL. OpenAI used it for InstructGPT, Dota 2, and many robotics tasks.
