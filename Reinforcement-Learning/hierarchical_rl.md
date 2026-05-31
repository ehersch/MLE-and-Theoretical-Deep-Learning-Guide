# Hierarchical Reinforcement Learning

Hierarchical RL (HRL) addresses **temporal abstraction** — the idea that complex behaviors can be decomposed into high-level plans executed by low-level primitives. This is how humans think: "go to the store" → "drive to parking lot" → "turn left at the light" → individual motor commands.

---

## The Problem with Flat RL

Long-horizon tasks have exponential state spaces over time. A task requiring 1000 steps with branching factor 10 has $10^{1000}$ paths. Credit assignment over 1000 steps is brutal — which early action caused the reward 950 steps later?

```
Flat RL:
  s_0 → a_0 → s_1 → a_1 → ... → s_{999} → a_{999} → r (+1)
        ↑
  How do we know which action mattered?
  
Hierarchical RL:
  High-level: g_0 → g_1 → g_2 → ... → g_K  (subgoals, every ~100 steps)
  Low-level:  for each g_k: execute policy π_low(a|s, g_k)
              (shorter horizon → easier credit assignment)
```

---

## The Options Framework

**Options** (Sutton, Precup, Singh 1999) provide a formal language for temporally extended actions.

An option $\omega$ is a triple $(\mathcal{I}_\omega, \pi_\omega, \beta_\omega)$:
- $\mathcal{I}_\omega \subseteq \mathcal{S}$: **initiation set** — states where the option can start
- $\pi_\omega: \mathcal{S} \to \mathcal{A}$: **intra-option policy** — what to do while executing the option
- $\beta_\omega: \mathcal{S} \to [0,1]$: **termination condition** — probability of stopping in each state

**Example options for a maze:**

```
Option "go_to_room_B":
  Initiation: {states in room A adjacent to corridor}
  Policy:     navigate toward room B
  Termination: β=1 when in room B

Option "pick_up_key":
  Initiation: {states near key}
  Policy:     move toward key, pick up
  Termination: β=1 when holding key
```

The **meta-policy** $\pi_\Omega(ω|s)$ selects which option to execute. Once chosen, the option runs until termination.

**SMDP (Semi-MDP):** with options, the high-level process is a Semi-MDP — transitions happen at variable time intervals (option durations), not fixed steps. Bellman equations for SMDPs account for the expected duration:

$$Q_\Omega(s, \omega) = \mathbb{E}\!\left[\sum_{t=0}^{T} \gamma^t r_t + \gamma^T \max_{\omega'} Q_\Omega(s_T, \omega')\right]$$

---

## HIRO: Hierarchical RL with Off-Policy Correction

HIRO (Nachum et al., 2018) is a practical HRL algorithm for continuous control with two levels:

```
┌─────────────────────────────────────────────────────────────┐
│                          HIRO                               │
│                                                             │
│  High-level policy (Manager):                               │
│    μ_hi(s_t) = g_t   (subgoal for next c steps)            │
│    Trained every c=10 steps                                 │
│    Reward: did the low-level achieve g_t over c steps?      │
│                                                             │
│  Low-level policy (Worker):                                 │
│    μ_lo(s_t, g_t) = a_t   (action to achieve subgoal)      │
│    Reward: r_lo = -‖s_{t+1} - g_t‖   (dense proximity)     │
│    Trained every step                                       │
│                                                             │
│  g_t: absolute state subgoal (target state for worker)      │
└─────────────────────────────────────────────────────────────┘
```

**The off-policy correction challenge:** the manager trained on transitions collected with an old low-level policy. But as the low-level improves, the same high-level action $g_t$ maps to different behaviors. 

HIRO fixes this by **relabeling** past high-level transitions with the subgoal that would have actually produced the observed low-level behavior under the current policy — analogous to HER for the hierarchical setting.

```python
def hiro_relabel_goal(s_t, s_t_c, candidate_goals, low_level_policy, actions_taken):
    """Find the subgoal g that best explains the observed low-level actions."""
    best_g, best_log_prob = None, -np.inf
    for g in candidate_goals:
        log_prob = sum(
            low_level_policy.log_prob(a, s, g)
            for s, a in zip(states_between, actions_taken))
        if log_prob > best_log_prob:
            best_log_prob, best_g = log_prob, g
    return best_g
```

---

## Subgoal Discovery

If subgoals aren't given, we need to discover them. **Bottleneck states** — states that must be traversed to get from one part of the state space to another — are natural subgoals.

```
Maze example:
  ┌─────┬─────┐
  │     │     │
  │  A  ╠══╣  B  │   ← doorway is a bottleneck state
  │     │     │
  └─────┴─────┘

Bottleneck detection:
  - States visited frequently on successful trajectories
  - Betweenness centrality in the transition graph
  - States where option policies terminate
```

**DADS (Dynamics-Aware Unsupervised Discovery of Skills):** learn skills $z$ such that the dynamics $P(s_{t+1}|s_t, z)$ are diverse and predictable across different skill values. Mutual information between $z$ and future states:

$$\mathcal{L}_{\text{DADS}} = I(s_{t+1}; z \mid s_t) = H(s_{t+1}|s_t) - H(s_{t+1}|s_t, z)$$

Skills that lead to reliably different outcomes are maximally useful for downstream tasks.

---

## Hierarchical RL for Language and Planning

HRL naturally maps to language tasks:

```
Goal: "Write a comprehensive report on climate change"

High-level (Manager): 
  Decompose into sections: ["Introduction", "Causes", "Effects", "Solutions"]

Mid-level:
  For each section: plan outline, gather information

Low-level:
  Generate each paragraph, sentence by sentence
```

Modern LLM agents often implement implicit hierarchical planning:
- **Chain-of-thought** = explicit low-level reasoning trace
- **Plan → execute** = two-level hierarchy
- **ReAct** = interleaved high-level thoughts + low-level actions

---

## Summary

```
Options framework: formal temporal abstraction (initiation, policy, termination)
HIRO: practical continuous HRL with relabeled subgoals
DADS: unsupervised skill discovery via dynamics diversity

When to use HRL:
  ✓ Long-horizon tasks (>200 steps)
  ✓ Compositional structure (subtasks that reuse skills)
  ✓ Sparse rewards across long horizons
  ✗ Short tasks (overhead not worth it)
  ✗ No natural temporal decomposition
```
