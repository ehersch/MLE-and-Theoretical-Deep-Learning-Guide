# Dynamic Programming

Dynamic programming (DP) solves MDPs **exactly** when you know the full model ($P$ and $R$). It's impractical for large problems but essential for building intuition about what all model-free algorithms are approximating.

---

## The Big Idea

The Bellman equations define $V^*$ implicitly. DP turns them into an iterative algorithm: start with an arbitrary guess $V_0$, apply the Bellman operator repeatedly, converge to $V^*$.

Why does this work? The Bellman operator is a **contraction** — it always moves $V$ closer to $V^*$.

---

## The Bellman Operator

Define the Bellman expectation operator $\mathcal{T}^\pi$ for a fixed policy $\pi$:

$$(\mathcal{T}^\pi V)(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V(s')\right]$$

And the Bellman optimality operator $\mathcal{T}^*$:

$$(\mathcal{T}^* V)(s) = \max_a \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V(s')\right]$$

**Contraction property (in $\ell_\infty$ norm):**

$$\|\mathcal{T}^* V - \mathcal{T}^* U\|_\infty \leq \gamma \|V - U\|_\infty$$

Since $\gamma < 1$, repeated application of $\mathcal{T}^*$ contracts the distance to the fixed point $V^*$ by $\gamma$ each step. After $k$ iterations, the error is $\gamma^k \|V_0 - V^*\|_\infty$.

---

## Policy Evaluation

**Problem:** given policy $\pi$, compute $V^\pi$.

**Algorithm:** iterate the Bellman expectation operator until convergence.

```
Initialize V(s) = 0 for all s
Repeat until convergence:
    For each s:
        V(s) ← Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
```

This is a system of $|\mathcal{S}|$ linear equations — we could solve it directly (for small $|\mathcal{S}|$) or iterate (scales better).

**Convergence:** $\|V_k - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty$

```python
import numpy as np

def policy_evaluation(pi, P, R, gamma, tol=1e-6):
    """
    pi: (S, A) array — pi[s, a] = probability of action a in state s
    P:  (S, A, S) array — P[s, a, s'] = transition probability
    R:  (S, A, S) array — R[s, a, s'] = reward
    """
    S = pi.shape[0]
    V = np.zeros(S)
    while True:
        V_new = np.einsum('sa,sas->s', pi, P * (R + gamma * V[None, None, :]))
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
```

---

## Policy Improvement

**Key insight:** given $V^\pi$, we can always find a policy that is at least as good.

**Greedy improvement:** for each state, take the action that maximizes the Q-value under $V^\pi$:

$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^\pi(s')\right]$$

**Policy improvement theorem:** $V^{\pi'}(s) \geq V^\pi(s)$ for all $s$.

Why? The greedy policy is at least as good as the original at every step. Formally (abbreviated):

$$V^\pi(s) \leq Q^\pi(s, \pi'(s)) \leq V^{\pi'}(s)$$

---

## Policy Iteration

Alternate between policy evaluation and policy improvement:

```
Initialize π arbitrarily

Repeat:
    1. Policy Evaluation: compute V^π (iterate Bellman)
    2. Policy Improvement: π ← greedy(V^π)
Until policy doesn't change → π = π*
```

```
π_0 →[eval]→ V^{π_0} →[improve]→ π_1 →[eval]→ V^{π_1} →[improve]→ π_2 → ... → π*
```

**Convergence:** guaranteed in finite steps (at most $|\mathcal{A}|^{|\mathcal{S}|}$ possible policies, strictly improves each step).

**Cost per iteration:** $O(|\mathcal{S}|^2 |\mathcal{A}|)$ for matrix solve, repeated until convergence.

```python
def policy_iteration(P, R, gamma):
    S, A = P.shape[:2]
    pi = np.ones((S, A)) / A  # uniform policy
    while True:
        V = policy_evaluation(pi, P, R, gamma)
        # Greedy improvement
        Q = np.einsum('sas,sas->sa', P, R + gamma * V[None, None, :])
        pi_new = np.zeros((S, A))
        pi_new[np.arange(S), Q.argmax(axis=1)] = 1.0
        if np.all(pi_new == pi):
            return pi, V
        pi = pi_new
```

---

## Value Iteration

**Insight:** we don't need a fully converged $V^\pi$ before improving. Truncate policy evaluation to **one step** and combine with improvement:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V_k(s')\right]$$

This applies the Bellman optimality operator $\mathcal{T}^*$ once per step.

```
Initialize V_0(s) = 0 for all s

Repeat until ||V_{k+1} - V_k||_∞ < ε:
    V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V_k(s')]

Extract policy: π*(s) = argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V*(s')]
```

```python
def value_iteration(P, R, gamma, tol=1e-6):
    S, A = P.shape[:2]
    V = np.zeros(S)
    while True:
        Q = np.einsum('sas,sas->sa', P, R + gamma * V[None, None, :])
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            pi = np.zeros((S, A))
            pi[np.arange(S), Q.argmax(axis=1)] = 1.0
            return pi, V_new
        V = V_new
```

---

## Policy Iteration vs. Value Iteration

```
Policy Iteration:                  Value Iteration:
  Full V^π solve (expensive)         One Bellman backup (cheap)
  Few outer iterations                Many outer iterations
  Better when evaluation is cheap    Better when |S| is huge
  Converges in polynomial iters      Converges geometrically (rate γ)
```

**Generalized Policy Iteration (GPI):** the big-picture insight from Sutton & Barto. All RL algorithms can be seen as some form of interleaving evaluation (estimate V or Q for current policy) and improvement (update policy to be more greedy). The specific tradeoff varies.

```
     ┌──── Evaluation (makes V consistent with π) ────┐
     │                                                  │
     V^π ◄──────────────────────────────────── π       │
     │                                         ▲       │
     │   Improvement (makes π greedy w.r.t V) │       │
     └────────────────────────────────────────►┘       │
     │                                                  │
     V*  ◄──────────── (at convergence) ──────── π*    │
```

---

## Why DP Fails at Scale

The gridworld above has perhaps 100 states. A real robot has a continuous 12-dimensional state space. An Atari game has $256^{210 \times 160 \times 3}$ possible frames.

DP requires:
1. Explicit enumeration of all states
2. Knowledge of $P(s'|s,a)$ and $R(s,a,s')$
3. $O(|\mathcal{S}|^2 |\mathcal{A}|)$ per iteration

None of these hold in practice. This motivates **model-free RL** (learn $V^\pi$ or $Q^*$ from samples) and **function approximation** (represent $V$ with a neural network, not a table).
