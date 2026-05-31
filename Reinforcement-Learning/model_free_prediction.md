# Model-Free Prediction

**Problem:** estimate $V^\pi(s)$ from experience (sample trajectories), without knowing $P$ or $R$.

This is the "prediction" problem — we're not optimizing the policy yet, just evaluating it. This lets us understand the core mechanics before adding control.

---

## Monte Carlo Prediction

**Idea:** run full episodes, observe actual returns, use them as unbiased estimates of $V^\pi$.

```
Episode: s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T (terminal)

Observed return from state s_t:
    G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^{T-t} r_T

Update: V(s_t) ← V(s_t) + α(G_t - V(s_t))
```

```python
def monte_carlo_prediction(env, pi, gamma=0.99, alpha=0.01, n_episodes=1000):
    V = defaultdict(float)
    for _ in range(n_episodes):
        # Run full episode
        episode = []
        s = env.reset()
        done = False
        while not done:
            a = pi(s)
            s_next, r, done, _ = env.step(a)
            episode.append((s, r))
            s = s_next
        
        # Compute returns and update
        G = 0
        for s, r in reversed(episode):
            G = r + gamma * G
            V[s] += alpha * (G - V[s])
    return V
```

**Properties:**
- ✅ Unbiased: $\mathbb{E}[G_t] = V^\pi(s_t)$
- ✅ Works for episodic tasks
- ❌ High variance (return is sum of many random rewards)
- ❌ Must wait until episode end to update
- ❌ Doesn't work for continuing (non-episodic) tasks

---

## Temporal Difference (TD) Learning

**The core idea of TD:** we don't need the full return $G_t$ to update $V(s_t)$. We can **bootstrap** — use our current estimate of $V(s_{t+1})$ to form a target.

$$\text{TD target:} \quad y_t = r_t + \gamma V(s_{t+1})$$

$$V(s_t) \leftarrow V(s_t) + \alpha\underbrace{(r_t + \gamma V(s_{t+1}) - V(s_t))}_{\text{TD error } \delta_t}$$

Update happens **after every step**, not at episode end.

```python
def td0_prediction(env, pi, gamma=0.99, alpha=0.1, n_episodes=1000):
    V = defaultdict(float)
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            a = pi(s)
            s_next, r, done, _ = env.step(a)
            # TD update
            td_error = r + gamma * V[s_next] - V[s]
            V[s] += alpha * td_error
            s = s_next
    return V
```

**The TD error $\delta_t$** is a prediction error: how much did our value estimate change given the new information $(r_t, s_{t+1})$?

---

## MC vs TD: The Bias-Variance Tradeoff

This is one of the most important distinctions in RL.

```
                MC                         TD(0)
         ┌──────────────┐           ┌──────────────┐
Bias     │    None      │           │  Yes (uses   │
         │  (uses true  │           │  bootstrap   │
         │   return)    │           │  estimate)   │
         └──────────────┘           └──────────────┘
Variance │    High      │           │     Low      │
         │  (sum of     │           │  (single     │
         │  T rewards)  │           │   step)      │
         └──────────────┘           └──────────────┘
Updates  │ Episode end  │           │  Every step  │
         └──────────────┘           └──────────────┘
Works on │  Episodic    │           │ Episodic or  │
         │  only        │           │ Continuing   │
         └──────────────┘           └──────────────┘
```

**Intuition for bias:** TD uses $V(s_{t+1})$ which is an estimate. If our estimate is wrong, TD is biased. MC uses actual returns — no bias.

**Intuition for variance:** MC's return $G_t = r_t + \gamma r_{t+1} + \ldots$ is a sum of many random variables. Each one adds variance. TD only looks one step ahead.

---

## n-step Returns

A spectrum between MC (full return) and TD (1-step return):

$$G_t^{(n)} = r_t + \gamma r_{t+1} + \ldots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$$

```
n=1:  y = r_t + γV(s_{t+1})                    ← TD(0), low variance, biased
n=2:  y = r_t + γr_{t+1} + γ²V(s_{t+2})
n=∞:  y = r_t + γr_{t+1} + ... + γ^T r_T       ← Monte Carlo, unbiased, high variance

 Bias ──────────────────────────────────────────►
 Variance ◄─────────────────────────────────────
                        n
```

Choosing $n$ is a hyperparameter. Often $n=5$ or $n=10$ works better than either extreme.

---

## TD(λ) and Eligibility Traces

Instead of picking one $n$, take a **geometric average** of all $n$-step returns:

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$$

The weight on the $n$-step return decays geometrically with $\lambda$:
- $\lambda=0$: pure TD(0)
- $\lambda=1$: pure Monte Carlo
- $0 < \lambda < 1$: blend with exponentially decreasing weight on longer returns

```
Weight on n-step return:
                ▲
  (1-λ)·λ⁰     │█
  (1-λ)·λ¹     │ ██
  (1-λ)·λ²     │   ███
  (1-λ)·λ³     │     ████
                └──────────────► n
```

**Eligibility traces** allow efficient online computation of TD(λ). Instead of waiting for the full return, we maintain a trace $e(s)$ for each state that tracks "how recently and frequently was this state visited?":

$$e_t(s) = \gamma\lambda \, e_{t-1}(s) + \mathbf{1}[s_t = s]$$

Update all states simultaneously at each step:

$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s) \quad \text{for all } s$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

**Intuition:** states visited recently have high traces → they get updated more. The trace decays with $\gamma\lambda$ — states fade from memory as time passes.

```python
def td_lambda_prediction(env, pi, gamma=0.99, lam=0.9, alpha=0.01, n_episodes=1000):
    V = defaultdict(float)
    for _ in range(n_episodes):
        e = defaultdict(float)   # eligibility traces
        s = env.reset()
        done = False
        while not done:
            a = pi(s)
            s_next, r, done, _ = env.step(a)
            delta = r + gamma * V[s_next] - V[s]
            e[s] += 1.0  # accumulate trace
            for state in e:
                V[state]  += alpha * delta * e[state]
                e[state]  *= gamma * lam
            s = s_next
    return V
```

---

## Summary: The Prediction Algorithms

```
MC ──────────────────────────────────── TD(0)
   │                                      │
   │   λ=1           λ=0.5     λ=0        │
   │    │               │        │        │
   │    ▼               ▼        ▼        │
   │   Full          Blend    Bootstrap   │
   │  return                  1 step      │
   │                                      │
High variance                        Low variance
Unbiased                             Biased (until convergence)
Episodic only                        Episodic + continuing
```

In deep RL, TD methods dominate because:
1. They update every step (data efficient)
2. Low variance helps gradient-based optimization
3. Work for continuing tasks
