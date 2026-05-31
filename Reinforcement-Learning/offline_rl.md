# Offline Reinforcement Learning

In standard RL, the agent interacts with the environment to collect data. In **offline RL** (also called batch RL), the agent must learn entirely from a **fixed dataset** with no further environment interaction.

```
Online RL:
  Agent ──act──► Environment ──observe──► Agent (repeat)
  Data improves as policy improves ✓

Offline RL:
  Fixed dataset D = {(s, a, r, s')}
  Agent must learn from D alone — no new interactions
  Data quality is fixed forever
```

---

## Why Offline RL?

Real-world scenarios where online interaction is costly or dangerous:

```
Healthcare:  Can't experiment with patient treatments
Robotics:    Collecting robot data is slow and expensive
Autonomous driving: Can't learn by crashing real cars
Finance:     Can't run live experiments with money
```

You often have large logs of historical behavior (from humans or a prior policy) — offline RL turns that data into a better policy.

---

## The Core Challenge: Distributional Shift

Why not just apply Q-learning to the offline dataset?

**The extrapolation problem:**

Q-learning updates:
$$Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a')$$

The $\max_{a'} Q(s', a')$ queries Q at **out-of-distribution (OOD) actions** — actions that may never appear in the dataset.

```
Dataset actions:          Learned Q surface:
                              ↑ Q
  a ∈ {a₁, a₂, a₃}           │         ← erroneously high
                              │    ╭──╮  (OOD actions)
                              │   /    \ 
                              │──/──────\──────► a
                                a₁  a₂  a₃
                                  (in dataset)
```

Neural networks will confidently extrapolate high Q-values to unvisited regions. The policy chases these phantom high-Q actions → **distribution shift** → catastrophic failure.

---

## Conservative Q-Learning (CQL)

**Idea:** penalize Q-values at OOD actions. Learn a Q-function that is **conservative** — it provides a lower bound on the true Q-values for OOD actions.

The CQL objective:

$$\mathcal{L}_{\text{CQL}}(\phi) = \underbrace{\mathbb{E}_{\text{TD loss}}}_{\text{standard Bellman}} + \alpha \underbrace{\left(\mathbb{E}_{s\sim D}\left[\log\sum_a \exp Q_\phi(s,a)\right] - \mathbb{E}_{(s,a)\sim D}[Q_\phi(s,a)]\right)}_{\text{conservative penalty}}$$

**What the penalty does:**
- First term: pushes up $\log \sum_a \exp Q(s,a)$ = softmax over all actions ≈ pushes up Q at sampled (possibly OOD) actions
- Second term: pushes down Q at **dataset actions**
- Net effect: Q-values at OOD actions are penalized relative to dataset actions

```
Without CQL:     With CQL:
     Q                  Q
     │                  │
     │    ╭──╮          │
     │   /    \         │────────────
     │──/──────\──►     │──/\────────►
        data  OOD          data  OOD
              high                 low
```

This conservative Q-function makes the greedy policy prefer dataset actions, naturally avoiding OOD behavior.

```python
def cql_loss(Q_net, batch, alpha=1.0, gamma=0.99):
    s, a, r, s_next, done = batch
    
    # Standard TD loss
    with torch.no_grad():
        q_next = Q_net(s_next).max(1).values
        targets = r + gamma * q_next * (1 - done)
    q_vals = Q_net(s).gather(1, a.unsqueeze(1)).squeeze()
    td_loss = F.mse_loss(q_vals, targets)
    
    # CQL penalty: E[logsumexp Q(s, ·)] - E[Q(s, a_dataset)]
    logsumexp_q = torch.logsumexp(Q_net(s), dim=1).mean()
    q_data = q_vals.mean()
    cql_penalty = logsumexp_q - q_data
    
    return td_loss + alpha * cql_penalty
```

---

## IQL: Implicit Q-Learning

**Insight behind IQL (Kostrikov et al., 2021):** instead of penalizing OOD actions, avoid querying Q at OOD actions entirely.

Q-learning update: $y = r + \gamma \max_{a'} Q(s', a')$ ← queries all $a'$

IQL replaces $\max_{a'} Q(s', a')$ with the expectile of the Q-value distribution over dataset actions:

$$V(s) = \arg\min_V \mathbb{E}_{a \sim \pi_\beta}\!\left[L_\tau^\text{exp}(Q(s,a) - V(s))\right]$$

$$L_\tau^\text{exp}(u) = \begin{cases} \tau |u| & u \geq 0 \\ (1-\tau)|u| & u < 0 \end{cases}$$

With $\tau \to 1$, $V(s) \to \max_{a} Q(s,a)$ but estimated **using only dataset actions**. This sidesteps OOD entirely.

**Three-network IQL:**
```
1. Q-network:  Bellman backup using V(s')
2. V-network:  Expectile regression on Q values
3. Policy:     Advantage-weighted behavior cloning
```

Policy extraction via **advantage-weighted regression (AWR)**:

$$\pi(a|s) \propto \pi_\beta(a|s) \cdot \exp\!\left(\frac{Q(s,a) - V(s)}{\beta}\right)$$

This upweights good actions from the dataset without querying OOD actions.

---

## Decision Transformer: Offline RL as Sequence Modeling

**A radically different approach:** forget Bellman equations entirely. Treat the trajectory as a sequence and train a Transformer to predict actions conditioned on desired future returns.

**Return-conditioned generation:**

```
Input:  [R_to_go_0, s_0, a_0, R_to_go_1, s_1, a_1, ..., R_to_go_t, s_t]
Output: a_t

At test time: set R_to_go_0 = high desired return
              model generates actions to achieve that return
```

```
Trajectory: ──────────────────────────────────────────
            (r_1, s_1, a_1), (r_2, s_2, a_2), ...

Replace r_t with return-to-go: R_t = Σ_{k=t}^T r_k

Feed [(R_0, s_0, a_0), (R_1, s_1, a_1), ...] to GPT
Predict a_t given past context
```

```python
# Decision Transformer forward pass (sketch)
class DecisionTransformer(nn.Module):
    def __init__(self, obs_dim, action_dim, n_layers=6, d_model=128, context_len=20):
        super().__init__()
        # Embed returns, states, actions separately
        self.return_embed = nn.Linear(1, d_model)
        self.state_embed  = nn.Linear(obs_dim, d_model)
        self.action_embed = nn.Linear(action_dim, d_model)
        # GPT-style transformer
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=8), n_layers)
        self.action_head = nn.Linear(d_model, action_dim)
    
    def forward(self, returns, states, actions):
        # Interleave: (R_0, s_0, a_0, R_1, s_1, a_1, ...)
        R_emb = self.return_embed(returns.unsqueeze(-1))
        s_emb = self.state_embed(states)
        a_emb = self.action_embed(actions)
        # Stack and flatten: T × 3 × d → 3T × d
        seq = torch.stack([R_emb, s_emb, a_emb], dim=2).flatten(1, 2)
        out = self.transformer(seq, seq)  # causal masking
        # Predict action from state tokens (every 3rd position + 1)
        return self.action_head(out[:, 1::3])
```

**Advantages:**
- No Bellman equations, no bootstrapping, no instability
- Can leverage large pretrained transformers
- Easy to add new modalities (images, language)

**Disadvantages:**
- Needs high-return trajectories in the dataset (can't generalize beyond dataset quality much)
- Doesn't "stitch" suboptimal trajectories together (Q-learning can combine a good first half + good second half from different trajectories)

---

## Offline RL Algorithm Comparison

```
                 CQL        IQL       DT
Avoids OOD?      Yes        Yes       Yes (no Bellman)
Bootstrapping    Yes        Yes       No
Stitching?       Yes        Yes       No (limited)
Stable?          Moderate   High      Very high
Implementation   Medium     Medium    Simple
Best when        Dense data Mixed     High-quality demos
```

**Practical advice:** IQL is often the best first try. CQL when you have dense reward and want policy improvement beyond behavior cloning. Decision Transformer when you have clean expert demos and want simplicity.
