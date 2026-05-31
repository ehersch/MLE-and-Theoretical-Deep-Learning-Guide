# Model-Based Reinforcement Learning

Model-free RL is general but sample-inefficient — millions of environment interactions are needed to learn simple locomotion tasks. **Model-based RL** (MBRL) learns a model of the environment and uses it to plan or generate synthetic data, dramatically improving sample efficiency.

---

## The Sample Efficiency Gap

```
                Task: MuJoCo HalfCheetah (continuous locomotion)

  SAC (model-free):   ~1,000,000 environment steps to solve
  MBPO (model-based): ~100,000 environment steps to solve  ← 10× better
  Human (analogy):    ~10 attempts to learn to ride a bike ← uses a "world model"
```

Why? Humans don't learn from scratch each trial — we have a mental model of physics, infer causality, plan ahead. MBRL mimics this.

---

## The Dyna Architecture

The earliest MBRL framework (Sutton, 1990). A simple but powerful idea: learn a model, use it to generate "imagined" experience to augment real experience.

```
┌──────────────────────────────────────────────────────────┐
│                        DYNA                              │
│                                                          │
│  Real env ──► replay buffer ──► model-free update (Q)    │
│      ↑                   ↑                               │
│      │              imagined transitions                  │
│      │         (from learned model M(s,a) → s', r)       │
│      └──────── real transitions ──► train model M        │
└──────────────────────────────────────────────────────────┘
```

**Algorithm:**
```
For each step:
  1. Take real action, observe (s, a, r, s')
  2. Update model: M_φ ← M_φ + gradient on (s, a) → (s', r)
  3. For k imagined steps:
     - Sample (s, a) from replay buffer
     - Generate (s', r) = M_φ(s, a)
     - Update Q-function on imagined transition
```

With $k=50$ imagined steps per real step, you get 50× better data efficiency at the cost of model computation. **Real environment steps are precious; compute is cheap.**

---

## Learning the Dynamics Model

**Gaussian neural network model:**

$$P(s_{t+1}|s_t, a_t) = \mathcal{N}(\mu_\phi(s_t, a_t), \Sigma_\phi(s_t, a_t))$$

Learn mean and variance jointly (heteroscedastic). Uncertainty-aware predictions.

**Ensemble models:** train $N=5$ independent networks, use disagreement as uncertainty:

```python
class EnsembleDynamicsModel:
    def __init__(self, n_models=5, obs_dim=17, act_dim=6):
        self.models = nn.ModuleList([
            DynamicsNet(obs_dim, act_dim) for _ in range(n_models)])
    
    def predict(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        predictions = [m(sa) for m in self.models]
        means = torch.stack([p[0] for p in predictions])
        stds  = torch.stack([p[1] for p in predictions])
        # Epistemic uncertainty = variance across ensemble means
        epistemic_var = means.var(0)
        return means.mean(0), stds.mean(0), epistemic_var
    
    def loss(self, s, a, s_next):
        total_loss = 0
        for model in self.models:
            mu, log_var = model(torch.cat([s, a], -1))
            # Gaussian NLL
            total_loss += (0.5 * log_var + (s_next - mu)**2 / (2 * log_var.exp())).mean()
        return total_loss
```

**Delta prediction:** predict $\Delta s = s_{t+1} - s_t$ instead of $s_{t+1}$ directly. This tends to be easier (smaller magnitude, more stationary).

---

## MBPO: Model-Based Policy Optimization

MBPO (Janner et al., 2019) integrates the ensemble model with SAC using **short model rollouts** from real states.

**Key insight:** long model rollouts accumulate error (each step's error compounds). Short rollouts from real starting states stay close to the true distribution.

```
MBPO Algorithm:

For each real step:
  1. Collect 1 real transition → add to real buffer D_env
  2. Update dynamics model on D_env
  3. For each of M model rollouts:
     - Sample s_0 from D_env (real state)
     - Roll out H steps using model: s_0, a_0, s_1', a_1, ..., s_H'
     - Add synthetic data to model buffer D_model
  4. Update SAC policy using mix of D_env ∪ D_model
```

**Short horizon H:** MBPO uses $H=1$–$5$. With just 1-step rollouts, you effectively learn a better reward + transition signal but avoid accumulating model errors.

```
Error analysis:
  Model error per step: ε_m
  Compound error over H steps: ε_m · H

  Real data: 0 error but scarce
  H=1 synthetic: ε_m error, 50x more data
  H=10 synthetic: 10·ε_m error, 50x more data → bad!
```

MBPO achieves SAC-level final performance but with **10-50× fewer real environment steps**.

---

## World Models and Dreamer

**World models** learn a compact **latent** representation of the environment, then plan and learn entirely in that latent space — never unrolling the full observation-space dynamics.

```
Observation s_t → Encoder → Latent state z_t → Decoder → s_t (reconstructed)
                                  │
                            Latent dynamics:
                            z_{t+1} = f(z_t, a_t)  ← fast! small vectors
```

**DreamerV2/V3 (Hafner et al., 2020-2023):**

Three learned components:
1. **World model:** learns $z_{t+1} = f(z_t, a_t)$ and $r_t = g(z_t, a_t)$ in latent space
2. **Actor:** policy $\pi_\theta(a|z)$ trained on **imagined** rollouts in latent space
3. **Critic:** value function $V_\phi(z)$ trained with TD in latent space

```
┌────────────────────────────────────────────────────────────┐
│                    Dreamer Training Loop                   │
│                                                            │
│  Real env → collect episodes → store in replay buffer      │
│                                                            │
│  World model training:                                     │
│    real images → encoder → z → RSSM → z' → decoder        │
│    minimize reconstruction loss + reward prediction loss   │
│                                                            │
│  Behavior learning (ENTIRELY in imagination):              │
│    z_0 (from buffer) → imagined rollout using world model  │
│    actor/critic learn from imagined returns                │
│                                                            │
│  Real env step: act using actor(z) (z from encoder)       │
└────────────────────────────────────────────────────────────┘
```

**RSSM (Recurrent State Space Model):** the latent dynamics combine a recurrent component (for memory) with a stochastic component (for uncertainty):

$$h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) \quad \text{(deterministic GRU)}$$
$$z_t \sim q(z_t | h_t, o_t) \quad \text{(stochastic, from observation)}$$
$$z_t \sim p(z_t | h_t) \quad \text{(prior, for imagination)}$$

**DreamerV3 achievements:**
- Mastered 150+ tasks spanning video games, robotics, locomotion, and more — with a single set of hyperparameters
- Collected diamonds in Minecraft from scratch (a hard exploration task)
- Requires only 1M environment steps where model-free methods need 100M+

---

## AlphaZero as MBRL

AlphaZero is MBRL in disguise: it uses the **known** game rules as a perfect model and plans with MCTS. No learned dynamics needed — the model is given.

This represents the ideal end of MBRL: if you have a perfect model (or a near-perfect simulator), use it for planning. If not, learn the model. (See [mcts_and_planning.md](mcts_and_planning.md))

---

## Tradeoffs

```
              Model-free          Model-based
Sample eff.   Low (millions       High (thousands
              of steps)           of steps)
Final perf.   High (no model      Can be limited by
              error)              model error
Compute       Per step: cheap     Training model:
                                  expensive
Best for      Simulation-cheap   Real-world robotics,
              problems           expensive simulators
```

**The model-error problem:** a learned model is always wrong to some degree. Policy optimization on a wrong model can find adversarial policies that exploit model errors. Short rollouts (MBPO) and conservative planning (with uncertainty) mitigate this.
