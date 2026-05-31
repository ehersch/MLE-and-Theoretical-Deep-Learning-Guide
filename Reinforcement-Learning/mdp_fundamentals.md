# MDP Fundamentals

Before any algorithm, we need the mathematical framework. The **Markov Decision Process (MDP)** is the universal formalism for sequential decision-making under uncertainty.

---

## Why MDPs?

Imagine a robot navigating a maze, or an algorithm deciding which ad to show, or a language model generating the next token. All of these share a structure:

- There is some **state** of the world
- The agent takes an **action**
- The world transitions to a new state and emits a **reward**
- This repeats

The "Markov" in MDP means: **the future depends only on the present state, not the full history**.

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

This is the **Markov property**. It's why we can talk about a "state" at all — the state is exactly the information needed to predict the future.

---

## Formal Definition

An MDP is a tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \mu_0)$:

| Symbol | Name | Meaning |
|--------|------|---------|
| $\mathcal{S}$ | State space | All possible situations the agent can be in |
| $\mathcal{A}$ | Action space | All possible actions the agent can take |
| $P(s' \mid s, a)$ | Transition dynamics | Probability of landing in $s'$ after taking $a$ in $s$ |
| $R(s, a, s')$ | Reward function | Scalar feedback signal |
| $\gamma \in [0, 1)$ | Discount factor | How much to weight future rewards |
| $\mu_0$ | Initial state distribution | Where episodes start |

---

## Discount Factor $\gamma$

Why discount at all? Three reasons:

1. **Mathematical:** ensures infinite-horizon returns converge: $\sum_{t=0}^\infty \gamma^t r_t \leq \frac{r_{\max}}{1 - \gamma}$
2. **Practical:** a reward now is worth more than the same reward later (uncertainty, opportunity cost)
3. **Behavioral:** $\gamma$ controls how far-sighted the agent is

```
γ = 0.99:  agent cares about rewards 100 steps away at ~37% of immediate reward
γ = 0.9:   agent cares about rewards 10 steps away at ~35% of immediate reward
γ = 0.0:   purely greedy (myopic) agent
```

The **return** (total discounted reward) from time $t$:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^\infty \gamma^k r_{t+k}$$

Note the recursive structure: $G_t = r_t + \gamma G_{t+1}$. This is the seed of the Bellman equations.

---

## Policy

A **policy** $\pi$ tells the agent what to do in each state.

**Deterministic:** $\pi: \mathcal{S} \to \mathcal{A}$, i.e., $a = \pi(s)$

**Stochastic:** $\pi: \mathcal{S} \to \Delta(\mathcal{A})$, i.e., $a \sim \pi(\cdot | s)$

Why stochastic? Sometimes randomization is optimal (e.g., rock-paper-scissors, partially observable environments) and it's needed for exploration.

---

## Value Functions

The **state-value function** $V^\pi(s)$ is the expected return starting from $s$ and following $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{t=0}^\infty \gamma^t r_t \;\middle|\; s_0 = s\right]$$

The **action-value function** $Q^\pi(s, a)$ conditions on taking action $a$ first, then following $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi\!\left[\sum_{t=0}^\infty \gamma^t r_t \;\middle|\; s_0 = s, a_0 = a\right]$$

**Relationship:**
$$V^\pi(s) = \sum_a \pi(a|s)\, Q^\pi(s, a)$$
$$Q^\pi(s, a) = \mathbb{E}_{s'}\!\left[R(s,a,s') + \gamma V^\pi(s')\right]$$

---

## The Bellman Equations

The Bellman equations express value functions **recursively**. They are the central equations of RL.

### Bellman Expectation Equation (for $V^\pi$)

$$\boxed{V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^\pi(s')\right]}$$

In words: the value of a state equals the expected immediate reward plus the discounted expected value of the next state, averaged over actions (under $\pi$) and transitions.

```
         π(a|s)          P(s'|s,a)
    s ──────────► a ───────────────► s'
    ↑                                ↓
    V^π(s)          r + γV^π(s')
```

### Bellman Expectation Equation (for $Q^\pi$)

$$Q^\pi(s, a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')\right]$$

### Bellman Optimality Equations

The **optimal value functions** $V^*$ and $Q^*$ satisfy:

$$\boxed{V^*(s) = \max_a \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^*(s')\right]}$$

$$Q^*(s, a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')\right]$$

The key difference: $\sum_a \pi(a|s)$ (expectation under $\pi$) becomes $\max_a$ (greedy over all actions).

---

## Optimal Policy

Given $Q^*$, the optimal policy is simply:

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

**Key theorem:** There always exists a deterministic optimal policy for a finite MDP. You never need randomness to be optimal (in fully-observed MDPs).

---

## Example: Gridworld

```
┌───┬───┬───┬───┐
│ S │   │   │ G │   S = Start, G = Goal (+1)
├───┼───┼───┼───┤
│   │ X │   │ X │   X = Wall/Hole (-1)
├───┼───┼───┼───┤
│   │   │   │   │
└───┴───┴───┴───┘

Actions: {up, down, left, right}
Reward: +1 at G, -1 at X, -0.01 everywhere else (step cost)
γ = 0.9
```

The optimal policy should:
- Navigate toward G efficiently
- Avoid X states
- Prefer shorter paths (step cost + discounting)

The Bellman equations form a system of $|\mathcal{S}|$ equations in $|\mathcal{S}|$ unknowns ($V^*(s)$ for each state). For small MDPs, we solve them exactly. For large/continuous MDPs, we approximate — that's the rest of the course.

---

## The RL Taxonomy

```
                    Know P(s'|s,a)?
                   /               \
                 YES                NO
                  |                  |
          Model-Based RL       Model-Free RL
         (Dynamic Programming)  /           \
                           Value-Based    Policy-Based
                           (Q-learning)  (Policy Gradients)
                                  \           /
                                  Actor-Critic
```

**Model-based:** compute $V^*$ or $\pi^*$ using the known dynamics (DP). See [dynamic_programming.md](dynamic_programming.md).

**Value-based model-free:** learn $Q^*$ from experience, derive policy greedily. See [model_free_control.md](model_free_control.md).

**Policy-based model-free:** directly optimize $\pi_\theta$. See [policy_gradients.md](policy_gradients.md).
