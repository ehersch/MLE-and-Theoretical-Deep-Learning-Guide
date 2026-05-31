# Reinforcement Learning

A comprehensive guide to modern reinforcement learning — from tabular MDPs through deep RL, offline RL, and RL for large language models. Inspired by Stanford CS224R and CS234.

---

## The RL Problem in One Picture

```
            ┌─────────────────────────────────┐
            │           Environment           │
            │                                 │
            │   s_{t+1}, r_t = f(s_t, a_t)   │
            └──────────┬──────────────────────┘
                       │  state s_t, reward r_t
                       ▼
            ┌──────────────────────┐
            │        Agent         │
            │  π(a | s) → action   │
            └──────────────────────┘
                       │  action a_t
                       └──────────────────────┐
                                              ▼
                                       Environment...
```

The agent observes state $s_t$, takes action $a_t \sim \pi(\cdot|s_t)$, receives reward $r_t$, transitions to $s_{t+1}$. Goal: find $\pi$ that maximizes cumulative reward.

---

## What makes RL different from supervised learning?

| Supervised Learning | Reinforcement Learning |
|---|---|
| Labels given at every step | Reward is sparse, delayed, and noisy |
| IID data | Sequential, correlated data |
| Static dataset | Data distribution depends on policy |
| No exploration needed | Must explore to discover rewards |

---

## Contents

| File | Topics |
|------|--------|
| [mdp_fundamentals.md](mdp_fundamentals.md) | MDP formalism, Bellman equations, value functions |
| [dynamic_programming.md](dynamic_programming.md) | Policy/value iteration, contraction mappings |
| [bandits_and_exploration.md](bandits_and_exploration.md) | Multi-armed bandits, UCB, Thompson sampling, regret |
| [model_free_prediction.md](model_free_prediction.md) | MC, TD(0), TD(λ), eligibility traces |
| [model_free_control.md](model_free_control.md) | SARSA, Q-learning, DQN, Double DQN |
| [policy_gradients.md](policy_gradients.md) | REINFORCE, policy gradient theorem, GAE |
| [actor_critic.md](actor_critic.md) | A2C, TRPO, PPO |
| [off_policy_actor_critic.md](off_policy_actor_critic.md) | SAC, TD3, entropy regularization |
| [offline_rl.md](offline_rl.md) | CQL, IQL, Decision Transformer, distributional shift |
| [imitation_learning.md](imitation_learning.md) | Behavioral cloning, DAgger, GAIL |
| [reward_learning.md](reward_learning.md) | Preference-based RL, reward modeling, Goodhart's law |
| [model_based_rl.md](model_based_rl.md) | Dyna, MBPO, world models, Dreamer |
| [exploration.md](exploration.md) | Count-based, curiosity, RND, Go-Explore |
| [multi_task_and_goal_conditioned.md](multi_task_and_goal_conditioned.md) | HER, goal-conditioning, UVFAs |
| [meta_rl.md](meta_rl.md) | RL², MAML, in-context RL |
| [hierarchical_rl.md](hierarchical_rl.md) | Options, HIRO, subgoal discovery |
| [mcts_and_planning.md](mcts_and_planning.md) | UCT, AlphaZero, MuZero |
| [rl_for_llms.md](rl_for_llms.md) | Token MDPs, PPO for LMs, GRPO, RLVR |
| [robotics_rl.md](robotics_rl.md) | Sim-to-real, domain randomization, VLAs |

## Reading order

**Foundations:** mdp_fundamentals → dynamic_programming → bandits_and_exploration → model_free_prediction → model_free_control

**Deep RL:** policy_gradients → actor_critic → off_policy_actor_critic

**Advanced:** offline_rl → imitation_learning → reward_learning → model_based_rl → exploration

**Special topics:** multi_task_and_goal_conditioned → meta_rl → hierarchical_rl → mcts_and_planning

**Applications:** rl_for_llms → robotics_rl
