# Monte Carlo Tree Search and Planning

MCTS is a family of best-first tree search algorithms that combine tree search with Monte Carlo sampling. Combined with deep neural networks (AlphaGo/AlphaZero), it achieved superhuman performance in board games long considered AI-hard.

---

## Why Not Just Use DP for Games?

Chess has ~$10^{44}$ legal positions. Full minimax tree search is impossible. Classic solutions (alpha-beta pruning, evaluation heuristics) were brittle. MCTS takes a different approach: **sample promising paths** rather than exhaustively searching.

---

## Basic MCTS

MCTS builds a search tree by repeatedly running four phases:

```
             ROOT (current state)
            / │ \
           /  │  \
          A   B   C         ← tried actions
         / \
        A1  A2              ← next level

Four phases, repeated N times:

1. SELECTION: traverse tree from root using UCT until leaf node
2. EXPANSION: add one or more children to the leaf
3. SIMULATION: random rollout from new node → terminal state
4. BACKPROPAGATION: propagate result up through visited nodes
```

### UCT: Upper Confidence Bound for Trees

At each node, select the child that maximizes:

$$\text{UCT}(s, a) = \underbrace{\frac{W(s,a)}{N(s,a)}}_{\text{exploitation}} + c\sqrt{\underbrace{\frac{\ln N(s)}{N(s,a)}}_{\text{exploration}}}$$

- $W(s,a)$: total value of action $a$ from state $s$ across all simulations
- $N(s,a)$: visit count for $(s,a)$
- $N(s)$: visit count for state $s$
- $c$: exploration constant

Same UCB formula from bandits, applied at every tree node!

```python
import math
from collections import defaultdict

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.W = 0.0  # total value
        self.N = 0    # visit count
        self.is_expanded = False
    
    def ucb(self, c=1.41):
        if self.N == 0: return float('inf')
        return self.W / self.N + c * math.sqrt(math.log(self.parent.N) / self.N)

def mcts(root_state, env, n_simulations=1000):
    root = MCTSNode(root_state)
    
    for _ in range(n_simulations):
        # 1. SELECTION: traverse to leaf
        node = root
        while node.is_expanded and node.children:
            node = max(node.children, key=lambda n: n.ucb())
        
        # 2. EXPANSION
        if not node.is_expanded:
            for action in env.legal_actions(node.state):
                s_next = env.step(node.state, action)
                node.children.append(MCTSNode(s_next, parent=node, action=action))
            node.is_expanded = True
            node = node.children[0]  # visit first child
        
        # 3. SIMULATION (random rollout)
        s = node.state
        done = False
        while not done:
            a = env.random_action(s)
            s, r, done = env.step(s, a)
        value = r  # terminal reward
        
        # 4. BACKPROPAGATION
        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent
    
    # Return action with most visits (most robust)
    return max(root.children, key=lambda n: n.N).action
```

---

## AlphaGo → AlphaZero

**AlphaGo** (DeepMind, 2016): first to beat a professional Go player. Key idea: replace the random rollout (simulation phase) with a **value network** and guide selection with a **policy network**.

**AlphaZero** (2017): simplified, generalized version that learned from scratch (no human knowledge) and dominated chess, shogi, and Go.

### Two Networks

$$\text{Neural network } f_\theta(s) = (p(s), v(s))$$

- $p(s) \in \Delta(\mathcal{A})$: **policy prior** — which moves are promising?
- $v(s) \in [-1, 1]$: **value estimate** — who's winning from this position?

### AlphaZero MCTS

Modify the UCT formula to include the policy prior:

$$\text{PUCT}(s, a) = \frac{W(s,a)}{N(s,a)} + c \cdot p(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

The policy prior $p(s,a)$ steers exploration toward promising moves even before they've been tried.

**Simulation:** instead of random rollout to terminal state, just evaluate $v(s_{\text{leaf}})$ from the value network. Much faster and more accurate.

### Self-Play Training Loop

```
┌──────────────────────────────────────────────────────┐
│              AlphaZero Training Loop                 │
│                                                      │
│  1. Self-play with current network f_θ               │
│     At each state: run N=800 MCTS simulations        │
│     Move probabilities: π ∝ N(s,a)^{1/τ}            │
│     Play until terminal → result z ∈ {-1, 0, +1}    │
│                                                      │
│  2. Store training data: (s, π, z)                   │
│                                                      │
│  3. Train network f_θ to minimize:                   │
│     L = (v(s) - z)²  [value loss]                    │
│       - π·log(p(s))  [policy loss]                   │
│       + λ‖θ‖²         [regularization]               │
│                                                      │
│  4. Repeat → stronger network → harder self-play     │
└──────────────────────────────────────────────────────┘
```

**The virtuous cycle:**

```
Better network → better self-play → harder games → better training data
                      ↑_______________________________________________│
```

AlphaZero trained for 9 hours (chess) and surpassed Stockfish, which had decades of human-engineered tuning.

---

## MuZero: Planning Without a Known Model

**AlphaZero limitation:** requires the full game rules (environment model) to simulate states during MCTS. What about environments where you don't know the rules?

**MuZero** (Schrittwieser et al., 2020): learn a latent dynamics model, and plan in latent space.

Three learned functions:
```
Representation:  h(s_0) → z_0          (encode observation to latent state)
Dynamics:        g(z_t, a_t) → (z_{t+1}, r_t)  (latent transitions + reward)
Prediction:      f(z_t) → (p_t, v_t)   (policy + value from latent state)
```

MCTS operates entirely in latent space using $g$ for transitions and $f$ for value/policy estimates. No environment model needed.

```
Real state s_0 → h → z_0 → [MCTS using g, f] → action
                                 │
                        Imagined rollouts in z-space
                        (no need for real game rules)
```

**Training:** match latent-space predictions to actual observed rewards and returns over $K$ unrolled steps.

MuZero matched or exceeded AlphaZero in board games, and also worked on Atari (no ground-truth game rules) with same set of hyperparameters.

---

## MCTS + LLMs

MCTS is increasingly used with LLMs for reasoning:

**Tree-of-Thought (ToT):** expand the reasoning tree with multiple thought candidates at each step, evaluate with a value model, use MCTS to search the best chain of thought.

```
Problem statement (root)
  ├── Approach A: "Use calculus..."
  │     ├── Step A1: "Differentiate f(x)..."   ← promising (v=0.8)
  │     └── Step A2: "Substitute x=0..."
  └── Approach B: "Use algebra..."             ← not promising (v=0.2)
        └── ...

Value model: is this intermediate reasoning step on track to a correct answer?
MCTS guides generation toward high-value reasoning paths.
```

AlphaProof (DeepMind, 2024) used MCTS over formal proof steps (Lean) to solve International Mathematical Olympiad problems at silver-medal level.
