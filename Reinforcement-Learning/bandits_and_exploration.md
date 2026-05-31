# Bandits and Exploration

The **multi-armed bandit** is the simplest RL problem: no state, no transitions — just choose an action, get a reward. It isolates the exploration-exploitation tradeoff in its purest form.

---

## Motivation: The Casino Problem

```
  Slot machine 1: μ₁ = 0.3   ← pull arm 1
  Slot machine 2: μ₂ = 0.7   ← pull arm 2 (actually best!)
  Slot machine 3: μ₃ = 0.5   ← pull arm 3

You don't know μ₁, μ₂, μ₃. You have 1000 pulls. How do you find the best
machine AND maximize total reward?
```

The tension:
- **Exploit:** pull the arm with the highest *estimated* reward so far
- **Explore:** pull arms you haven't tried much to get better estimates

Exploit too much → miss the true best arm. Explore too much → waste pulls on bad arms.

This is the exploration-exploitation tradeoff, and it shows up everywhere in RL.

---

## Problem Setup

$K$ arms with unknown reward distributions. At each step $t$:
- Choose arm $a_t \in \{1, \ldots, K\}$
- Receive reward $r_t \sim \mathcal{D}_{a_t}$ with mean $\mu_{a_t}$

Let $\mu^* = \max_k \mu_k$ (best arm's mean). Define **regret**:

$$R_T = T\mu^* - \sum_{t=1}^T \mu_{a_t} = \sum_{t=1}^T (\mu^* - \mu_{a_t})$$

Regret = how much better we would have done if we always pulled the best arm. The goal is to **minimize $R_T$**.

---

## ε-Greedy

The simplest strategy: with probability $\varepsilon$ explore (random arm), with probability $1-\varepsilon$ exploit (best estimated arm).

```
Estimate Q̂(a) = mean of observed rewards for arm a

At each step:
    with prob ε:  a_t = uniform random arm
    with prob 1-ε: a_t = argmax_a Q̂(a)
```

```python
import numpy as np

class EpsilonGreedy:
    def __init__(self, k, eps=0.1):
        self.k, self.eps = k, eps
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
    
    def select(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.k)
        return np.argmax(self.values)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        # Incremental mean update
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
```

**Regret:** $O(T)$ — linear. No matter how many pulls you have, you keep exploring at rate $\varepsilon$. With $\varepsilon_t = 1/t$ (decreasing), you get $O(\log T)$ regret, but the schedule is hard to tune.

---

## UCB: Upper Confidence Bound

A principled solution. The key insight: be **optimistic in the face of uncertainty**. Try arms that either have high estimated value OR haven't been tried much.

$$a_t = \arg\max_a \underbrace{\hat{\mu}_a}_{\text{exploit}} + \underbrace{c\sqrt{\frac{\ln t}{N_a}}}_{\text{explore bonus}}$$

where $N_a$ = number of times arm $a$ has been pulled, $c$ is a constant (often $\sqrt{2}$).

**Intuition of the bonus:**

```
Confidence interval for μ_a:
    
    ───────────[─────────────────────────────]───
           μ̂_a - bonus          μ̂_a + bonus
    
    Wide interval (few pulls): UCB is high → explore
    Narrow interval (many pulls): UCB ≈ μ̂_a → exploit
```

Each arm's bonus shrinks as we pull it more. The algorithm naturally balances exploration and exploitation.

```python
class UCB:
    def __init__(self, k, c=np.sqrt(2)):
        self.k, self.c = k, c
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.t = 0
    
    def select(self):
        self.t += 1
        # Pull each arm once first
        for a in range(self.k):
            if self.counts[a] == 0:
                return a
        ucb = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return np.argmax(ucb)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
```

**Regret bound (UCB1):**

$$R_T \leq \sum_{a: \mu_a < \mu^*} \frac{8 \ln T}{\Delta_a} + \left(1 + \frac{\pi^2}{3}\right)\sum_{a} \Delta_a$$

where $\Delta_a = \mu^* - \mu_a$ is the gap. This is $O(\sqrt{KT \ln T})$ in the worst case — **sub-linear regret**! The algorithm is asymptotically optimal up to constants (Lai-Robbins lower bound: $\Omega(\log T)$).

---

## Thompson Sampling (Bayesian Bandit)

A beautifully simple Bayesian approach. Maintain a **posterior** over each arm's reward mean. At each step, sample from the posterior and pick the arm with the highest sample.

For Bernoulli rewards (0 or 1), use a Beta prior:
- Prior: $\mu_a \sim \text{Beta}(\alpha_a, \beta_a)$ (initialized as $\text{Beta}(1,1)$ = uniform)
- Update: after reward $r$, update $\alpha_a \mathrel{+}= r$, $\beta_a \mathrel{+}= (1-r)$
- Select: sample $\tilde{\mu}_a \sim \text{Beta}(\alpha_a, \beta_a)$, pick $\arg\max_a \tilde{\mu}_a$

```python
class ThompsonSampling:
    def __init__(self, k):
        self.alpha = np.ones(k)  # successes + 1
        self.beta  = np.ones(k)  # failures + 1
    
    def select(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm]  += 1 - reward
```

**Intuition:**

```
Early (high uncertainty):
  Beta(1,1) ──────────── Beta(1,1)
  Both flat → random exploration

After arm 2 shows wins:
  Beta(1,1) ──── Beta(7,2)
  Arm 2 posterior peaked right → usually selected
```

Thompson Sampling has the same $O(\log T)$ regret as UCB but often performs better in practice. It's widely used in recommendation systems.

---

## Exploration in Full RL

Bandits = no state. In full RL, exploration is harder because:

1. **Credit assignment:** a reward received 20 steps after an action — which action caused it?
2. **State coverage:** the agent might never visit certain states if it only exploits

The same three strategies generalize:
- **ε-greedy:** random action with prob ε at each step
- **UCB/Optimism:** maintain uncertainty estimates for $Q(s,a)$ — hard with neural networks
- **Thompson Sampling / Posterior:** Bayesian Q-learning — computationally expensive

For deep RL, more scalable exploration methods are needed. See [exploration.md](exploration.md).

---

## Contextual Bandits

A middle ground between bandits and full RL: at each step you observe a **context** (feature vector) and choose an action. Reward depends on context and action. No state transitions.

$$a_t = \pi(x_t), \quad r_t \sim \mathcal{D}(x_t, a_t)$$

Applications: news article recommendation (context = user features), clinical treatment (context = patient features), ad selection.

The key challenge: learn a policy that generalizes across contexts. Linear models (LinUCB) or neural networks can parameterize the policy and uncertainty estimates.

---

## Summary

| Algorithm | Strategy | Regret | Key idea |
|-----------|----------|--------|----------|
| ε-greedy | Random explore | $O(T)$ | Simple, suboptimal |
| UCB1 | Optimism | $O(\sqrt{KT \ln T})$ | Confidence bounds |
| Thompson Sampling | Posterior sampling | $O(\sqrt{KT \ln T})$ | Bayesian uncertainty |

The lesson that carries into all of RL: **you must explore to learn, but exploration costs reward**. Good algorithms quantify uncertainty and explore where it's highest.
