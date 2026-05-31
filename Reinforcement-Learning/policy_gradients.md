# Policy Gradients

Value-based methods (Q-learning, DQN) find a policy indirectly: learn $Q^*$, then act greedily. **Policy gradient methods** directly optimize the policy $\pi_\theta$ by gradient ascent on expected return.

---

## Why Direct Policy Optimization?

Value-based methods have fundamental limitations:

1. **Discrete actions only (mostly):** $\arg\max_a Q(s,a)$ is intractable for continuous $a$
2. **Deterministic policies:** argmax gives a deterministic policy. Some problems require stochastic policies (partially observable, adversarial)
3. **Aliased states:** if the function approximator can't distinguish states, a deterministic policy may be suboptimal; stochastic is better

With policy gradients, we directly parameterize $\pi_\theta(a|s)$ (e.g., a softmax neural net) and optimize $\theta$ with gradient ascent.

---

## The Objective

We want to maximize the **expected return**:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^T \gamma^t r_t\right] = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$$

where $\tau = (s_0, a_0, r_0, s_1, \ldots)$ is a full trajectory sampled by running $\pi_\theta$.

We want $\nabla_\theta J(\theta)$ to do gradient ascent. The challenge: the expectation is over trajectories that **depend on $\theta$** — we can't just push the gradient inside.

---

## The Policy Gradient Theorem

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

This is the **policy gradient theorem**. Let's derive it.

### Derivation

$$J(\theta) = \sum_\tau P(\tau; \theta) G(\tau)$$

$$\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta P(\tau; \theta) G(\tau)$$

Use the **log-derivative trick**: $\nabla_\theta P = P \cdot \nabla_\theta \log P$

$$= \sum_\tau P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) \cdot G(\tau)$$

$$= \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\nabla_\theta \log P(\tau; \theta) \cdot G(\tau)\right]$$

Now, the trajectory probability:

$$P(\tau; \theta) = \mu_0(s_0) \prod_{t=0}^T P(s_{t+1}|s_t, a_t) \cdot \pi_\theta(a_t|s_t)$$

Taking the log and differentiating:

$$\nabla_\theta \log P(\tau; \theta) = \underbrace{\nabla_\theta \log \mu_0(s_0)}_{=0} + \sum_t \underbrace{\nabla_\theta \log P(s_{t+1}|s_t, a_t)}_{=0 \text{ (no }\theta)} + \nabla_\theta \log \pi_\theta(a_t|s_t)$$

Everything involving the environment dynamics drops out! We're left with:

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]}$$

**Intuition:** increase the log-probability of actions that led to high returns. Decrease it for low-return actions.

The **score function** $\nabla_\theta \log \pi_\theta(a_t|s_t)$ tells us which direction to move $\theta$ to make action $a_t$ more likely. We scale it by $G_t$ — take bigger steps toward actions that paid off.

---

## REINFORCE Algorithm

The simplest policy gradient algorithm: estimate the gradient using Monte Carlo returns.

```
Initialize θ arbitrarily
For each episode:
    Run policy π_θ: (s_0, a_0, r_0, ..., s_T)
    Compute returns: G_t = Σ_{k=t}^T γ^{k-t} r_k
    For each step t:
        θ ← θ + α · ∇_θ log π_θ(a_t|s_t) · G_t
```

```python
import torch, torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return torch.distributions.Categorical(logits=self.net(x))

def reinforce(env, n_episodes=2000, gamma=0.99, lr=1e-3):
    pi = PolicyNet(env.observation_space.shape[0], env.action_space.n)
    opt = optim.Adam(pi.parameters(), lr=lr)
    
    for _ in range(n_episodes):
        states, actions, rewards = [], [], []
        s = env.reset()
        done = False
        while not done:
            dist = pi(torch.FloatTensor(s))
            a = dist.sample()
            s_next, r, done, _ = env.step(a.item())
            states.append(s); actions.append(a); rewards.append(r)
            s = s_next
        
        # Compute returns
        G, returns = 0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        
        # Policy gradient update
        loss = 0
        for s, a, G in zip(states, actions, returns):
            dist = pi(torch.FloatTensor(s))
            loss -= dist.log_prob(a) * G  # negative because we do gradient ascent
        
        opt.zero_grad(); loss.backward(); opt.step()
```

---

## The Variance Problem

REINFORCE has very high variance. The return $G_t$ includes rewards from time $t$ onward, but also from before time $t$ — which action $a_t$ cannot have influenced!

**Causality:** action $a_t$ only affects rewards $r_t, r_{t+1}, \ldots$ (not past rewards). We can use the **reward-to-go** instead of the full return:

$$\nabla_\theta J(\theta) = \mathbb{E}\!\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \underbrace{\sum_{k=t}^T \gamma^{k-t} r_k}_{\text{reward-to-go}}\right]$$

This is still an unbiased estimator but with lower variance.

---

## Baselines: Variance Reduction

We can subtract any **baseline** $b(s_t)$ from the return without introducing bias:

$$\nabla_\theta J(\theta) = \mathbb{E}\!\left[\sum_{t} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$$

**Proof of unbiasedness:** for any baseline $b(s_t)$ that doesn't depend on $a_t$:

$$\mathbb{E}_{a_t \sim \pi}\!\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t)\right] = b(s_t) \underbrace{\mathbb{E}_{a_t}\!\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\right]}_{= 0} = 0$$

(The expectation of the score function is always zero: $\mathbb{E}[\nabla \log \pi] = \nabla \mathbb{E}[1] = 0$.)

**Best baseline:** the state value $b(s_t) = V^\pi(s_t)$. This gives us the **advantage**:

$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

Advantage = how much better is action $a_t$ than the average action in state $s_t$?

$$\nabla_\theta J(\theta) = \mathbb{E}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^\pi(s_t, a_t)\right]$$

Advantages are centered around zero, making updates much more stable.

---

## Generalized Advantage Estimation (GAE)

How do we estimate $A^\pi(s_t, a_t)$ in practice? We need $V^\pi$, which we learn as a separate **critic** network.

**TD error as advantage estimate:**

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \approx A^\pi(s_t, a_t)$$

This is a 1-step estimate — low variance but biased (if $V$ is wrong).

**GAE (Schulman et al., 2015):** like TD(λ) but for advantages — exponentially weighted average of $n$-step advantage estimates:

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{k=0}^\infty (\gamma\lambda)^k \delta_{t+k}$$

- $\lambda=0$: $\hat{A}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ — low variance, high bias
- $\lambda=1$: $\hat{A}_t = G_t - V(s_t)$ — unbiased, high variance
- $0 < \lambda < 1$: bias-variance tradeoff

In practice: $\lambda = 0.95$, $\gamma = 0.99$ work well across many environments.

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    # values: list of V(s_t) estimates, including V(s_T+1) = 0
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages
```

---

## Summary

```
REINFORCE: high variance, no critic, simple
    + baseline: reduced variance
    + reward-to-go: less variance still
    + GAE: smooth advantage estimates
    + separate critic: → Actor-Critic (next section)

Key equation:
    θ ← θ + α · E[∇_θ log π_θ(a|s) · Â(s,a)]
```

Policy gradients are the foundation of PPO, TRPO, SAC, and GRPO — the algorithms powering modern LLM training.
