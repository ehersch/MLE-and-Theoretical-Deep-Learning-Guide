# Imitation Learning

In many settings, defining a reward function is hard — but we can observe expert behavior. **Imitation learning** (IL) learns a policy from demonstrations, without explicit rewards.

```
Expert (human, optimal planner, etc.)
    │
    │ demonstrations: τ = {(s₀,a₀), (s₁,a₁), ..., (s_T, a_T)}
    ▼
Imitation learning algorithm
    │
    ▼
Learned policy π_θ ≈ π_expert
```

---

## Behavioral Cloning (BC)

**The simplest idea:** treat it as supervised learning. The expert's $(s_t, a_t)$ pairs are training data.

$$\mathcal{L}_{\text{BC}}(\theta) = -\mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{expert}}}\!\left[\log \pi_\theta(a|s)\right]$$

```python
def behavioral_cloning(expert_data, obs_dim, action_dim, n_epochs=100):
    policy = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                           nn.Linear(256, 256), nn.ReLU(),
                           nn.Linear(256, action_dim))
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    states, actions = zip(*expert_data)
    S = torch.FloatTensor(states)
    A = torch.FloatTensor(actions)
    for _ in range(n_epochs):
        loss = F.mse_loss(policy(S), A)   # continuous actions
        opt.zero_grad(); loss.backward(); opt.step()
    return policy
```

**The compounding error problem:**

```
Expert trajectory:                 BC policy trajectory:

s_0 → s_1 → s_2 → s_3 → G        s_0 → s_1' → s_2'' → s_3''' → ?
       ↑
   small error here                     ↑ errors compound!
   (never seen in training)         each step drifts further from
                                    the expert distribution
```

At test time, the BC policy makes a small error → ends up in a state $s'$ not in the training data → makes a bigger error → distribution keeps shifting. Error compounds at rate $O(\epsilon T^2)$ where $\epsilon$ is single-step error rate.

**When BC works:** short horizons, lots of demos, precise expert behavior.

---

## DAgger: Dataset Aggregation

**Fix the distributional shift** by interactively querying the expert on states the learner visits.

```
Algorithm DAgger:
    Initialize: D = {} (empty dataset), π₁ = arbitrary policy

    For i = 1, 2, ..., N:
        1. Run policy π_i in environment → collect states {s₀, s₁, ..., s_T}
        2. Query EXPERT for action labels at each visited state
        3. Aggregate: D ← D ∪ {(s_t, a_t^expert)}
        4. Train π_{i+1} via supervised learning on D

    Return: best π_i
```

```python
def dagger(env, expert, n_iters=20, n_rollout=100):
    policy = PolicyNet(obs_dim, action_dim)
    dataset = []
    for i in range(n_iters):
        # Run current policy, collect states
        s = env.reset()
        for _ in range(n_rollout):
            a_policy = policy(torch.FloatTensor(s)).sample().item()
            # Query expert at EVERY state the learner visits
            a_expert = expert(s)
            dataset.append((s, a_expert))     # label with expert action
            s, _, done, _ = env.step(a_policy)  # but execute learner's action
            if done: s = env.reset()
        
        # Retrain on aggregated dataset
        train_bc(policy, dataset)
    return policy
```

**Why does DAgger work?** The dataset now covers states the *learner* visits, not just states the *expert* visits. The policy is trained on the same distribution it encounters at test time.

**Error guarantee:** DAgger reduces compounding error from $O(\epsilon T^2)$ to $O(\epsilon T)$ — linear instead of quadratic.

**Practical challenge:** you need interactive access to the expert (a human labeler or a simulator). Offline-only settings can't use DAgger.

---

## Inverse Reinforcement Learning (IRL)

Instead of directly mimicking actions, **recover the reward function** that makes the expert's behavior optimal.

```
Forward RL:   reward R → optimal policy π*
Inverse RL:   expert demonstrations → reward R → policy π
```

**Why recover rewards?**
- Rewards generalize better than policies across new situations
- You might want to transfer behavior to a different environment
- The reward is the "true" objective; the policy is just one expression of it

**Maximum Entropy IRL (Ziebart et al., 2008):** Among all possible reward functions, find the one under which the expert's demonstrated trajectories are **maximum entropy optimal** — as random as possible while still matching the expert's feature expectations.

$$R^* = \arg\max_R \sum_{\tau \in D_{\text{expert}}} \log P(\tau; R) - \lambda\|R\|^2$$

Fitting IRL requires solving the forward RL problem as an inner loop (to compute the partition function $Z(R)$). Expensive, but theoretically grounded.

---

## GAIL: Generative Adversarial Imitation Learning

GAIL (Ho and Ermon, 2016) bypasses reward recovery entirely. Formulate imitation as an **adversarial game** between:
- **Discriminator** $D_\psi$: tries to distinguish expert $(s,a)$ pairs from learner $(s,a)$ pairs
- **Policy (generator)** $\pi_\theta$: tries to fool the discriminator

```
Expert data: (s, a) ~ π_E  ──► D(s,a) → 1
Learner data: (s, a) ~ π_θ ──► D(s,a) → 0

Policy tries to make D think its (s,a) came from the expert.
```

The implicit reward signal: $r(s, a) = -\log(1 - D_\psi(s, a))$

**Training loop:**
```
Repeat:
    1. Collect rollout from π_θ
    2. Update discriminator:
       maximize E_{π_E}[log D(s,a)] + E_{π_θ}[log(1-D(s,a))]
    3. Update policy with PPO/TRPO using reward r(s,a) = -log(1-D(s,a))
```

```python
# GAIL training sketch
discriminator = Discriminator(obs_dim + action_dim)
policy = PPO_Agent(obs_dim, action_dim)

for iteration in range(n_iters):
    # Collect learner rollout
    learner_data = policy.collect_rollout(env)
    
    # Update discriminator
    expert_loss = -F.logsigmoid(discriminator(expert_data)).mean()
    learner_loss = -F.logsigmoid(-discriminator(learner_data)).mean()
    disc_loss = expert_loss + learner_loss
    disc_optimizer.zero_grad(); disc_loss.backward(); disc_optimizer.step()
    
    # Assign intrinsic rewards from discriminator
    rewards = -torch.log(1 - discriminator(learner_data).sigmoid() + 1e-8)
    
    # Update policy with these rewards (e.g., PPO)
    policy.update(learner_data, rewards)
```

**GAIL in practice:** works very well with enough expert data and stable training (GAN training can be finicky). Surpasses BC significantly on complex continuous control tasks.

---

## Comparison

```
                BC        DAgger     IRL        GAIL
Expert needed  Offline   Interactive Offline   Offline
Reward output  No        No          Yes        No
Scales to      Medium    Large       Small      Large
  complex tasks demos    envs        problems   envs
Error           O(εT²)   O(εT)      O(εT)     O(εT)
Implementation  Simple   Medium     Complex    Complex
```

**When to use what:**
- Simple task, lots of demos → **BC**
- Interactive expert available → **DAgger**
- Need transferable reward → **IRL**
- Complex task, stable training → **GAIL**
- No reward, offline large-scale → **Offline IL (BC + Decision Transformer)**
