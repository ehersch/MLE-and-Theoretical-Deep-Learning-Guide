# Multi-Task and Goal-Conditioned RL

Standard RL learns one task (one reward function). **Multi-task RL** and **goal-conditioned RL** train policies that generalize across tasks or goals — a prerequisite for general-purpose agents.

---

## Goal-Conditioned RL

Instead of a fixed reward, condition the policy on a **goal** $g$:

$$\pi(a | s, g) \quad \text{instead of} \quad \pi(a | s)$$

Reward is defined relative to the goal:

$$r(s, a, g) = \begin{cases} +1 & \text{if goal } g \text{ is achieved} \\ 0 & \text{otherwise} \end{cases}$$

**Example:** robot arm manipulation

```
Goal = "move block to position (0.5, 0.3)"
Policy: π(a | arm_state, block_state, goal=(0.5, 0.3))

Same policy, different goal:
"move block to position (0.1, 0.8)"
π(a | arm_state, block_state, goal=(0.1, 0.8))
```

The policy learns a general movement skill, not a specific task.

**Goal representations:**
- Target state vector (position, orientation)
- Target image (what the world should look like)
- Language instruction ("pick up the red cube")
- Desired state delta (move 10cm left)

---

## The Sparse Reward Problem

Goal-conditioned RL has notoriously sparse rewards: the agent only gets +1 when it reaches the goal, which might happen with near-zero probability randomly.

```
Robotic arm reaching task:
  Random policy: probability of reaching goal ≈ 0.0001
  Episodes: 10,000
  Expected successes: 1
  → Policy never gets reward → learns nothing
```

---

## Hindsight Experience Replay (HER)

**HER** (Andrychowicz et al., 2017) is one of the most elegant ideas in RL. Instead of treating failed episodes as useless, **relabel them with the goal the agent actually achieved.**

```
Episode: agent tries to reach goal g = (3, 4)
  s_0=(0,0) → s_1=(1,0) → s_2=(1,1) → s_3=(2,1) → ... → s_T=(2,3)
  
  With goal g=(3,4): all rewards = 0 (never reached)  ← useless?
  
  HER relabels: pretend the goal was g' = (2,3) = s_T
  Now: final transition (s_{T-1}, a_{T-1}, s_T) gets reward +1!
  
  The agent learns: "if my goal is (2,3), doing THIS sequence works!"
```

**The insight:** every trajectory, no matter how "failed," is a successful trajectory toward the actual state it ended in. HER makes every episode informative.

```python
def her_replay(episode, strategy='final', k=4):
    """
    episode: list of (s, a, r, s_next, g) tuples
    strategy: 'final' (relabel with final state) or 
              'future' (relabel with random future state in episode)
    """
    transitions = list(episode)
    
    # Original transitions
    relabeled = transitions.copy()
    
    for t, (s, a, r, s_next, g) in enumerate(episode):
        if strategy == 'final':
            virtual_goals = [episode[-1][3]]  # final s_next as goal
        elif strategy == 'future':
            # Sample k future states as virtual goals
            future_indices = np.random.randint(t, len(episode), size=k)
            virtual_goals = [episode[i][3] for i in future_indices]
        
        for g_virtual in virtual_goals:
            # Recompute reward: did we achieve g_virtual?
            r_virtual = float(np.allclose(s_next, g_virtual, atol=0.05))
            relabeled.append((s, a, r_virtual, s_next, g_virtual))
    
    return relabeled
```

**HER strategies:**
- `final`: relabel with episode's final state → simple, always one extra goal
- `future`: relabel with random future states in episode → more data, better coverage
- `episode`: relabel with random states from same episode

HER enabled robots to learn **dexterous manipulation with sparse rewards** that previously required dense reward engineering or demonstrations.

---

## Universal Value Function Approximators (UVFAs)

Generalize Q-functions to take the goal as input:

$$Q(s, a, g; \theta) \approx Q^*(s, a, g)$$

Optimal policy: $\pi^*(a|s, g) = \arg\max_a Q(s, a, g)$

**Architecture:**

```
s ──► embed_s ──┐
                ├──► concat ──► MLP ──► Q(s,a,g)
a ──► embed_a ──┤
                │
g ──► embed_g ──┘
```

The key benefit: a single neural network that generalizes across all goals in the goal space. Instead of training separate Q-functions for each goal, one UVFA handles everything.

---

## Multi-Task RL

In multi-task RL, tasks differ not just in goal but in the reward structure entirely:

$$\mathcal{T} = \{T_1, T_2, \ldots, T_K\}, \quad T_i = (\mathcal{S}, \mathcal{A}, P, R_i, \gamma)$$

Each task has a different reward function. The goal: train one policy $\pi(a|s, z_i)$ where $z_i$ is a task identifier.

**Task encodings:**
- One-hot: $z_i = e_i$ (simple, but doesn't generalize to new tasks)
- Task embedding: $z_i = \psi_\xi(T_i)$ learned representation
- Language: $z_i = $ "pick up the red cup" → enables zero-shot generalization to new task descriptions

**Gradient interference:** gradients from different tasks can conflict. Task A says "go left," task B says "go right." Naive multi-task training can be worse than single-task.

**Multi-task gradient surgery:** project gradients from conflicting tasks to be orthogonal, preventing interference.

---

## The Full Picture

```
Single-task RL:        π(a|s)       → one task, one reward
Goal-conditioned RL:   π(a|s, g)    → generalize over goals
Multi-task RL:         π(a|s, z)    → generalize over tasks
Language-conditioned:  π(a|s, text) → generalize over instructions
```

Goal-conditioned RL + language instructions + large pretrained models → the recipe for general robot policies and language-guided agents (see [robotics_rl.md](robotics_rl.md)).

---

## HER + SAC: Practical Implementation

```python
class HER_SAC:
    def __init__(self, obs_dim, goal_dim, action_dim):
        # Policy conditioned on (obs, goal)
        self.sac = SAC(obs_dim + goal_dim, action_dim)
        self.replay_buffer = []
    
    def store_episode(self, episode, k=4):
        """episode: list of (obs, action, reward, next_obs, goal)"""
        # Store original transitions
        for transition in episode:
            self.replay_buffer.append(transition)
        
        # HER: add relabeled transitions
        for t, (s, a, r, s_next, g) in enumerate(episode):
            if t == len(episode) - 1: continue
            future_t = np.random.randint(t+1, len(episode))
            g_virtual = episode[future_t][3]   # s_next of future step as goal
            r_virtual = self.compute_reward(s_next, g_virtual)
            self.replay_buffer.append((s, a, r_virtual, s_next, g_virtual))
    
    def act(self, obs, goal, deterministic=False):
        obs_goal = np.concatenate([obs, goal])
        return self.sac.act(obs_goal, deterministic=deterministic)
    
    def update(self):
        batch = random.sample(self.replay_buffer, 256)
        s, a, r, s_next, g = zip(*batch)
        # Concatenate obs with goal
        s_g = np.concatenate([s, g], axis=-1)
        sn_g = np.concatenate([s_next, g], axis=-1)
        self.sac.update_batch(s_g, a, r, sn_g)
```

HER + SAC is the standard baseline for robotic manipulation with sparse rewards. It turns failed reaching attempts into successful "reached this position" lessons — and learns manipulation in 1/100th the steps of dense-reward RL.
