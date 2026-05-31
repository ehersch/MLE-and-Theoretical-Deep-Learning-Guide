# Model-Free Control

Now we want to find the **optimal policy** without knowing the model. We combine the prediction ideas from the last section with the policy improvement idea from dynamic programming.

---

## Why Q-values, Not V-values?

With a model, we can do policy improvement from $V^\pi$:

$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

Without a model, we can't compute this — we don't know $P$ or $R$. But if we learn $Q^\pi(s,a)$ directly, improvement is trivial:

$$\pi'(s) = \arg\max_a Q^\pi(s, a) \quad \leftarrow \text{no model needed!}$$

This is why model-free control focuses on **action-value functions**.

---

## SARSA: On-Policy Control

SARSA updates Q-values using transitions $(s, a, r, s', a')$ — the tuple that gives SARSA its name.

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\underbrace{(r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))}_{\text{TD error}}$$

"On-policy" means $a_{t+1}$ is drawn from the **current policy** (including exploration). The Q-estimate reflects the value of the policy being executed.

```python
def sarsa(env, gamma=0.99, alpha=0.1, eps=0.1, n_episodes=5000):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    def eps_greedy(s):
        if np.random.random() < eps:
            return env.action_space.sample()
        return np.argmax(Q[s])
    
    for _ in range(n_episodes):
        s = env.reset()
        a = eps_greedy(s)
        done = False
        while not done:
            s_next, r, done, _ = env.step(a)
            a_next = eps_greedy(s_next)
            # SARSA update
            Q[s][a] += alpha * (r + gamma * Q[s_next][a_next] - Q[s][a])
            s, a = s_next, a_next
    return Q
```

**Convergence:** SARSA converges to $Q^*$ if the policy is GLIE (Greedy in the Limit with Infinite Exploration) — i.e., all $(s,a)$ visited infinitely often, and policy converges to greedy.

---

## Q-Learning: Off-Policy Control

Q-learning decouples the behavior policy (used to collect data) from the target policy (being optimized). The update always bootstraps with the **greedy** action:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\left(r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right)$$

```
SARSA:       uses Q(s', a') where a' ~ π(·|s')   ← what we'd actually do
Q-learning:  uses max_a' Q(s', a')               ← best possible action
```

Q-learning is **off-policy**: the behavior policy can be anything (even random!) as long as it covers all $(s,a)$ pairs. This is hugely useful — we can learn from any collected data.

```python
def q_learning(env, gamma=0.99, alpha=0.1, eps=0.1, n_episodes=5000):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            # Behavior policy: ε-greedy
            a = env.action_space.sample() if np.random.random() < eps else np.argmax(Q[s])
            s_next, r, done, _ = env.step(a)
            # Q-learning update: greedy target
            Q[s][a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s][a])
            s = s_next
    return Q
```

**SARSA vs Q-Learning in a cliff-walking environment:**

```
┌────────────────────────────┐
│ S                        G │
│                            │
│ ─ ─ ─ CLIFF ─ ─ ─ ─ ─    │ ← -100 if fallen
└────────────────────────────┘

Q-learning:  learns optimal path close to cliff (risky but best)
SARSA:       learns safer path away from cliff (accounts for ε-greedy noise)
```

SARSA is "safe" — it accounts for its own randomness. Q-learning finds the true optimum but can suffer during learning because the behavior policy still explores.

---

## Deep Q-Network (DQN)

Q-learning with a table doesn't scale. Replace the table with a neural network:

$$Q(s, a; \theta) \approx Q^*(s, a)$$

**Why not just apply Q-learning with a neural net?** It breaks catastrophically. Three problems:

1. **Correlated updates:** consecutive $(s_t, a_t, r_t, s_{t+1})$ transitions are highly correlated → gradient updates are correlated → training diverges (like training a supervised model where batch $k$ perfectly predicts what batch $k+1$ will look like)

2. **Non-stationary targets:** the TD target $r + \gamma \max_{a'} Q(s', a'; \theta)$ uses the same network $\theta$ we're updating → target moves as we train → instability

3. **Overestimation:** $\max_{a'} Q(s', a'; \theta)$ is positively biased because $\mathbb{E}[\max] \geq \max[\mathbb{E}]$

DQN (Mnih et al., 2015) fixes all three:

### Fix 1: Experience Replay

Store transitions $(s, a, r, s', \text{done})$ in a **replay buffer**. Sample random minibatches for updates.

```
Agent collects transition → push to replay buffer (circular queue)
                                         ↓
                          Sample random minibatch of 32-256 transitions
                                         ↓
                                   Gradient update
```

- Breaks temporal correlations (random sampling)
- Data is reused multiple times (sample efficiency)

### Fix 2: Target Network

Maintain a separate **target network** $\theta^-$ that is updated slowly:

$$\text{Loss} = \mathbb{E}\!\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

$\theta^-$ is frozen for $C$ steps, then copied from $\theta$. This makes the target stable while we optimize $\theta$.

```
  θ  (online net): updated every step by gradient descent
  θ⁻ (target net): frozen, copied from θ every C=1000 steps
```

### DQN Implementation

```python
import torch, torch.nn as nn, torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),     nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x): return self.net(x)

class DQN:
    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.99,
                 buffer_size=10000, batch_size=64, target_update=1000):
        self.n_actions = n_actions
        self.gamma, self.batch_size = gamma, batch_size
        self.online = QNetwork(obs_dim, n_actions)
        self.target = QNetwork(obs_dim, n_actions)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.steps = 0
        self.target_update = target_update
    
    def act(self, s, eps=0.1):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            return self.online(torch.FloatTensor(s)).argmax().item()
    
    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))
    
    def update(self):
        if len(self.buffer) < self.batch_size: return
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s_next, done = map(torch.FloatTensor, zip(*batch))
        a, done = a.long(), done.bool()
        
        # Current Q values
        q_vals = self.online(s).gather(1, a.unsqueeze(1)).squeeze()
        
        # Target Q values (frozen network)
        with torch.no_grad():
            next_q = self.target(s_next).max(1).values
            targets = r + self.gamma * next_q * (~done)
        
        loss = nn.MSELoss()(q_vals, targets)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        
        # Periodically copy online → target
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
```

---

## DQN Improvements

### Double DQN

The fix for overestimation: decouple action *selection* (online net) from action *evaluation* (target net):

$$\text{Target} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

Regular DQN: $r + \gamma \max_{a'} Q(s', a'; \theta^-)$ — same network selects and evaluates → bias

Double DQN: online net picks the action, target net scores it → much less bias.

### Dueling DQN

Decompose Q into state value + advantage:

$$Q(s, a; \theta) = V(s; \theta) + A(s, a; \theta) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a'; \theta)$$

```
          ┌─────────────┐
          │  Shared CNN │
          └──────┬──────┘
         ┌───────┴───────┐
         ▼               ▼
     V(s) head       A(s,·) head
    (1 output)     (|A| outputs)
         └───────┬───────┘
                 ▼
              Q(s,·)
```

**Why?** In many states, the choice of action doesn't matter much (e.g., empty room in Atari). Dueling separates "how good is this state?" from "which action is better here?" — more efficient learning.

### Prioritized Experience Replay

Sample transitions proportional to their **TD error** — surprising transitions are more informative:

$$P(\text{sample } i) \propto |\delta_i|^\alpha$$

High-error transitions (the model is most wrong about them) are replayed more. Use importance sampling weights to correct for the sampling bias.

---

## DQN: The Full Picture

```
Environment ──► Replay Buffer ──► Random minibatch
                                        │
                    Online Q-net ───────┤──► TD loss
                    Target Q-net ───────┘
                         ▲
                         │ copy every C steps
                    Online Q-net

Behavior: ε-greedy with online Q-net (ε anneals over training)
```

DQN achieved human-level performance on 49 Atari games from raw pixels — a landmark result. But it only works for discrete action spaces. For continuous actions, we need actor-critic methods.
