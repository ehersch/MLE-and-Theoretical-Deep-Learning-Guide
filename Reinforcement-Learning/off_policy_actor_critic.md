# Off-Policy Actor-Critic: SAC and TD3

PPO is on-policy: each batch of data is collected with the current policy and thrown away after one (or a few) gradient updates. This is sample-inefficient. **Off-policy** methods can reuse data from a replay buffer, drastically improving sample efficiency.

---

## The Maximum Entropy Framework

Standard RL maximizes expected return:
$$J(\pi) = \mathbb{E}\!\left[\sum_t r_t\right]$$

**Maximum entropy RL** adds an entropy bonus:

$$J_{\text{MaxEnt}}(\pi) = \mathbb{E}\!\left[\sum_t r_t + \alpha \underbrace{H(\pi(\cdot|s_t))}_{\text{entropy of policy}}\right]$$

where $H(\pi(\cdot|s)) = -\sum_a \pi(a|s)\log\pi(a|s)$ and $\alpha$ is the **temperature** parameter.

**Why maximize entropy?**

```
Low entropy policy (overfit):        High entropy policy:
   a=LEFT always in state s           multiple actions explored
   → misses other good actions        → robust, harder to exploit
   → brittle if dynamics change       → better exploration

Entropy term encourages visiting all actions proportional to their Q-value,
not just the single best one.
```

The maximum entropy objective leads to **soft** versions of Bellman equations where the optimal policy is a softmax over Q-values rather than a hard argmax.

---

## Soft Actor-Critic (SAC)

SAC (Haarnoja et al., 2018) is the most popular off-policy continuous control algorithm. Three components:

### 1. Soft Q-function (Critic)

Two Q-networks (twin critics) to reduce overestimation:

$$\mathcal{L}_Q(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}}\!\left[\left(Q_\phi(s,a) - y\right)^2\right]$$

$$y = r + \gamma \left(\min_{j=1,2} Q_{\phi_j'}(s', \tilde{a}') - \alpha \log \pi_\theta(\tilde{a}'|s')\right), \quad \tilde{a}' \sim \pi_\theta(\cdot|s')$$

where $\phi'$ are target network parameters (soft update: $\phi' \leftarrow \tau\phi + (1-\tau)\phi'$).

### 2. Policy (Actor)

The policy maximizes the soft Q-value:

$$\mathcal{L}_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{B}, a \sim \pi_\theta}\!\left[\alpha \log \pi_\theta(a|s) - \min_j Q_{\phi_j}(s, a)\right]$$

For continuous actions, $\pi_\theta$ is a Gaussian with learned mean and variance. We use the **reparameterization trick** to backpropagate through sampling:

$$a = \mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This makes the sample differentiable w.r.t. $\theta$.

### 3. Automatic Temperature Tuning

$\alpha$ balances exploration vs exploitation. SAC can automatically tune it by treating it as a dual variable:

$$\mathcal{L}(\alpha) = \mathbb{E}_{a \sim \pi}\!\left[-\alpha \log \pi(a|s) - \alpha \bar{H}\right]$$

where $\bar{H}$ is the target entropy (e.g., $-|\mathcal{A}|$ for continuous). This drives entropy to the target automatically.

### SAC Implementation

```python
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from collections import deque
import random

class GaussianPolicy(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -20, 2
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
    
    def forward(self, s):
        h = self.net(s)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        # Reparameterization trick
        eps = torch.randn_like(std)
        a_pre = mean + std * eps
        a = torch.tanh(a_pre)  # squash to [-1, 1]
        # Log prob of the squashed Gaussian
        log_prob = torch.distributions.Normal(mean, std).log_prob(a_pre)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6)
        return a, log_prob.sum(-1)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))
    def forward(self, s, a): return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)

class SAC:
    def __init__(self, obs_dim, action_dim, alpha=0.2, gamma=0.99, tau=0.005, lr=3e-4):
        self.gamma, self.tau, self.alpha = gamma, tau, alpha
        self.actor = GaussianPolicy(obs_dim, action_dim)
        self.critic1 = QNetwork(obs_dim, action_dim)
        self.critic2 = QNetwork(obs_dim, action_dim)
        self.target1 = QNetwork(obs_dim, action_dim)
        self.target2 = QNetwork(obs_dim, action_dim)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
        self.buffer = deque(maxlen=int(1e6))
    
    def update(self, batch_size=256):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = map(torch.FloatTensor, zip(*batch))
        
        with torch.no_grad():
            a_next, log_pi = self.actor(s_next)
            q1_next = self.target1(s_next, a_next)
            q2_next = self.target2(s_next, a_next)
            q_target = r + self.gamma * (1 - done) * (
                torch.min(q1_next, q2_next) - self.alpha * log_pi)
        
        # Critic update
        q1_loss = F.mse_loss(self.critic1(s, a), q_target)
        q2_loss = F.mse_loss(self.critic2(s, a), q_target)
        self.critic_opt.zero_grad(); (q1_loss + q2_loss).backward(); self.critic_opt.step()
        
        # Actor update
        a_new, log_pi = self.actor(s)
        q_min = torch.min(self.critic1(s, a_new), self.critic2(s, a_new))
        actor_loss = (self.alpha * log_pi - q_min).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        
        # Soft target update
        for target, source in [(self.target1, self.critic1), (self.target2, self.critic2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
```

---

## TD3: Twin Delayed Deep Deterministic Policy Gradient

TD3 (Fujimoto et al., 2018) builds on DDPG (a deterministic off-policy AC) with three improvements:

### 1. Twin Critics (clipped double Q)

Same as SAC: two critics, use the minimum for targets. Prevents Q-value overestimation.

### 2. Delayed Policy Updates

Update the actor less frequently than the critics (e.g., every 2 critic updates). The critics converge more before the actor uses them.

### 3. Target Policy Smoothing

Add noise to the target action (smooths the Q-function to prevent exploiting sharp Q peaks):

$$y = r + \gamma Q_{\phi'}(s', \text{clip}(\mu_{\theta'}(s') + \epsilon, -c, c))$$

$$\epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)$$

```
              SAC vs TD3 summary:
              ┌─────────────────────────────────────────────┐
              │              SAC            TD3             │
              │ Policy    Stochastic     Deterministic      │
              │           Gaussian       + noise at train   │
              │ Entropy   Explicit       Implicit (noise)   │
              │ Critics   Twin + min     Twin + min         │
              │ Target    Soft EMA       Soft EMA           │
              │ Best for  Most envs      Simpler reward     │
              │           High entropy   landscapes         │
              └─────────────────────────────────────────────┘
```

---

## When to Use Which Algorithm?

```
Discrete actions, simple env:     Q-learning / DQN
High-dimensional discrete:        DQN + Dueling/Double
On-policy, episodic:              PPO (+ GAE)
Off-policy, continuous:           SAC (default choice)
Off-policy, continuous, simpler:  TD3
Off-policy, discrete:             SAC with Gumbel-Softmax
LLM alignment:                    PPO-Clip / GRPO (see rl_for_llms.md)
```

SAC is often the first thing to try for continuous control — it's robust to hyperparameters, sample-efficient, and handles exploration automatically through entropy maximization.
