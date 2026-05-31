# Exploration in Deep RL

Exploration is trivial in bandits (just try each arm). In deep RL with huge state spaces, exploration is one of the hardest open problems. Some environments have **sparse rewards** — the agent can wander for millions of steps without encountering any reward signal.

---

## The Hard Exploration Problem

```
┌──────────────────────────────────────────┐
│            Montezuma's Revenge           │
│                                          │
│  ████████████████████████████████        │
│  █ Start         █             █        │
│  █    Agent      ████    ███   █        │
│  █         ████           ██   █        │
│  █                              █        │
│  █  Need to find KEY, open DOOR █        │
│  █  then find TREASURE          █        │
│  ██████████████████████████████████      │
│                                          │
│  Sparse reward: +0 until you collect     │
│  treasure (many rooms away)              │
│                                          │
│  ε-greedy DQN: never gets past room 1   │
│  Human: can solve it                     │
└──────────────────────────────────────────┘
```

This game stumped DQN for years. Humans solve it intuitively because we have curiosity — we're intrinsically motivated to explore novel situations.

---

## Count-Based Exploration

**Idea:** visit states you haven't seen much. Formally, add a bonus reward proportional to visit rarity.

For tabular MDPs, just count visits $N(s,a)$:

$$r_{\text{bonus}}(s, a) = \frac{c}{\sqrt{N(s, a)}}$$

This is exactly the UCB bonus from the bandit setting, applied to each $(s,a)$ pair.

**The problem at scale:** you can't count individual states in a 256×256 pixel game. Every frame is unique.

**Pseudo-count:** generalize counting to continuous/high-dimensional spaces using a **density model** $\rho_n(s)$ fitted on past experience:

$$\hat{N}(s) \approx \rho_n(s) \cdot n \quad \text{(pseudo-count)}$$

The density increases as we visit similar states → pseudo-count increases → bonus decreases. Works with neural density models.

---

## Curiosity-Driven Exploration (ICM)

**Intrinsic Curiosity Module (ICM)** (Pathak et al., 2017) gives the agent a reward for encountering states it can't predict well — novelty as a proxy for informativeness.

```
┌─────────────────────────────────────────────────────────────┐
│                          ICM                                │
│                                                             │
│  s_t ──► Feature encoder φ ──► φ(s_t)                      │
│  s_{t+1} ──► Feature encoder φ ──► φ(s_{t+1})              │
│                                                             │
│  Forward model: φ(s_t), a_t → φ̂(s_{t+1})                  │
│  Inverse model: φ(s_t), φ(s_{t+1}) → â_t                   │
│                                                             │
│  Intrinsic reward: r_i = ‖φ(s_{t+1}) - φ̂(s_{t+1})‖²       │
│                         ↑                                   │
│                    prediction error = surprise              │
└─────────────────────────────────────────────────────────────┘
```

**Why a feature encoder?** Raw pixels change trivially (leaves blowing, clouds). ICM learns features that are only about **action-relevant** state changes via the inverse model (which action caused this state transition?). Irrelevant changes are filtered out.

**Total reward:** $r_t = r_{\text{extrinsic}} + \beta r_{\text{intrinsic}}$

ICM was the first method to make significant progress on Montezuma's Revenge without demonstrations.

**The "noisy TV" problem:** if you put a TV with random noise in the environment, ICM gives infinite curiosity reward for watching it — it's always surprising. Pure curiosity can get "addicted" to unpredictable noise.

---

## RND: Random Network Distillation

**RND** (Burda et al., 2018) fixes the noisy TV problem with an elegant trick.

A randomly initialized, **fixed** target network $f: \mathcal{S} \to \mathbb{R}^k$. Train a predictor network $\hat{f}_\theta: \mathcal{S} \to \mathbb{R}^k$ to match it:

$$r_{\text{intrinsic}} = \|f(s) - \hat{f}_\theta(s)\|^2$$

**Why this works:**
- For novel states: predictor hasn't trained on them → high error → high bonus
- For visited states: predictor has seen similar states → low error → low bonus
- The fixed target has no temporal structure → noise TV gives constant, not growing, bonus

```python
class RND(nn.Module):
    def __init__(self, obs_dim, emb_dim=128):
        super().__init__()
        self.target = nn.Sequential(     # fixed random network
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, emb_dim))
        self.predictor = nn.Sequential( # trained to predict target
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, emb_dim))
        # Freeze target
        for p in self.target.parameters():
            p.requires_grad = False
    
    def intrinsic_reward(self, obs):
        with torch.no_grad():
            target_emb = self.target(obs)
        pred_emb = self.predictor(obs)
        return (pred_emb - target_emb).pow(2).mean(-1)  # per-state bonus
    
    def predictor_loss(self, obs):
        target_emb = self.target(obs).detach()
        pred_emb = self.predictor(obs)
        return (pred_emb - target_emb).pow(2).mean()
```

RND achieved state-of-the-art on Montezuma's Revenge with pure curiosity, surpassing methods that used demonstrations.

---

## Go-Explore

**Insight:** hard exploration problems fail because the agent **forgets** how to reach interesting states. Once a lucky exploration reaches a good state, it random-walks away and rarely returns.

Go-Explore (Ecoffet et al., 2019) separates exploration into two phases:

**Phase 1: Explore**
```
Maintain an archive of interesting states (e.g., novel game states)
For each iteration:
  1. Select a state from archive (e.g., random or by novelty)
  2. RESTORE environment to that state (deterministic reset)
  3. Explore from there (random actions for k steps)
  4. Add any new states to archive
```

**Phase 2: Robustify**
```
Once archive has high-scoring trajectories:
  Imitation learn a policy that can reach those states from scratch
  (without the ability to reset to arbitrary states)
```

```
Archive:
┌──────────────────────────────────────┐
│  State A (room 1, score 0)   ← found │
│  State B (room 2, score 100) ← found │
│  State C (room 3, score 300) ← found │
└──────────────────────────────────────┘
Go-explore: "let me start from State C and explore further"
→ finds State D (room 4, score 400)
```

Go-Explore achieved superhuman performance on Montezuma's Revenge (score 43,000 vs human ~4,700). Key requirement: the simulator must support deterministic state restoration (which game simulators do).

---

## Exploration Strategy Comparison

```
Environment type    Recommended strategy
─────────────────────────────────────────────────────
Dense rewards       ε-greedy / entropy bonus (PPO)
Moderate sparsity   RND or ICM bonus
Hard sparse (games) Go-Explore
Real world robot    Uncertainty-based (ensemble models)
Combinatorial (NLP) Diverse sampling, temperature > 1
```

---

## Exploration in LLMs

For LLM RLHF and GRPO, exploration means generating **diverse** outputs rather than always sampling the same high-probability response.

**Approaches:**
- **Temperature > 1:** flatten output distribution → more diverse sampling
- **Entropy bonus:** reward the policy for maintaining high entropy (same as SAC's entropy term)
- **Diversity rewards:** penalize generating the same response multiple times for the same prompt
- **Best-of-N:** generate many samples, take the best — pure exploration at inference time

The "hard exploration" problem in LLMs is discovering **new reasoning strategies** — analogous to finding keys in Montezuma's Revenge. RLVR (see [rl_for_llms.md](rl_for_llms.md)) solves this by training on verifiable rewards that only fire when the model discovers correct solutions.
