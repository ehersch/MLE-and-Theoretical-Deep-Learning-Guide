# RL for Robotics: Sim-to-Real and VLAs

Robotics is the ultimate test of RL: the environment is the physical world, data collection is slow and expensive, hardware can break, and mistakes are costly. This section covers the techniques that make RL practical for real robots.

---

## The Simulation Gap

Training RL policies in simulation is cheap — you can run millions of steps on a GPU cluster. But simulation is not reality:

```
Simulation                          Reality
───────────────────────────────────────────────────
Perfect physics                     Friction, flex, backlash
Exact sensor readings               Noise, latency, dropped packets
No wear / fatigue                   Parts degrade
Exact actuator response             Motor lag, hysteresis
Uniform lighting                    Shadows, glare, changing light
Clean background                    Clutter, people walking by
```

A policy trained purely in simulation often **fails completely** when deployed on a real robot. This is the **sim-to-real gap** (or reality gap).

---

## Domain Randomization

**Key insight:** if you randomize the simulation parameters widely enough, the real world is just another sample from the distribution. The policy must be robust to all variations → it generalizes to reality.

```
┌──────────────────────────────────────────────────────────────┐
│                    Domain Randomization                       │
│                                                              │
│  Randomize at each episode:                                   │
│    Physics:   friction ∈ [0.5, 1.5]                          │
│               mass ∈ [0.8, 1.2] × nominal                    │
│               joint damping ∈ [0.5, 2.0] × nominal          │
│    Visuals:   texture (random)                               │
│               lighting (position, intensity, color)          │
│               camera position (small perturbations)          │
│               object appearance (color, shininess)           │
│    Dynamics:  action delay ∈ {0, 1, 2} timesteps            │
│               observation noise σ ∈ [0, 0.02]               │
└──────────────────────────────────────────────────────────────┘

Trained policy must succeed across ALL of these → robust to real world
```

**OpenAI Dactyl (2019):** a robotic hand learned to solve a Rubik's cube using only domain randomization — never trained on the real hand, yet achieved dexterity that took years of engineering before.

**Tradeoffs:** too little randomization → policy overfits to simulation. Too much → task becomes too hard, policy learns overly conservative behavior.

---

## Adaptive Domain Randomization

Instead of manually tuning randomization ranges, automatically adjust them to maximize transfer.

**Automatic Domain Randomization (ADR):** start with a narrow randomization range. If the policy succeeds consistently → expand the range. If it fails consistently → shrink it.

```
difficulty ────────────────────────────────────────────────►
          │
          │  ADR threshold
  P(success) ▲
          │  ████████████
          │ ██          ██
          │██            ██
          └──────────────────►  range width
          
Expand range when P(success) > upper_threshold
Shrink range when P(success) < lower_threshold
```

**RMARL (Randomization-based Meta-RL):** train the policy as a meta-learner that adapts to the specific randomization setting in a few steps — analogous to meta-RL across tasks.

---

## System Identification and Adaptive Policies

Rather than ignoring the sim-real gap, **identify** the real system's parameters.

**RMA (Rapid Motor Adaptation) (Kumar et al., 2021):**

Phase 1: train a policy conditioned on privileged simulation parameters $e$ (friction, mass, etc.):

$$\pi(a | s, e)$$

Phase 2: train an **adaptation module** $\phi$ that estimates $e$ from the history of observations and actions:

$$\hat{e} = \phi(s_{t-H:t}, a_{t-H:t})$$

At deployment: use $\hat{e}$ from the adaptation module. The policy automatically adjusts for the real robot's properties.

```
Deployment:
  s_0, s_1, ..., s_t (observations)  →  adaptation module φ  →  ê_t
                                                                    ↓
  s_t  ─────────────────────────────────────────────────────► π(a|s_t, ê_t) → a_t
```

RMA enabled a quadruped robot to walk on ice, gravel, stairs, and through tall grass — all without any fine-tuning on the physical robot.

---

## Real-World Data Collection Challenges

When you must train (or fine-tune) on real hardware:

**Safe RL:** add constraints to prevent hardware damage:

$$\max_\pi J(\pi) \quad \text{s.t.} \quad \mathbb{E}[C(s,a)] \leq d_{\text{max}}$$

where $C(s,a)$ is a cost function (e.g., excessive torque, joint limits).

**Safety projection:** at each step, project actions to safe regions using a safety filter (CBF — Control Barrier Function):

```
RL action a_RL → Safety filter → Safe action a_safe → Robot
                       ↑
                  CBF constraint:
                  "if this action leads toward unsafe state,
                   project it back to safe region"
```

**Hardware-efficient RL:** techniques to minimize real-world interactions:
- Model-based RL (MBPO) for sample efficiency
- Offline RL on human demonstration data, then fine-tune with RL
- Short episodes (reset frequently)

---

## Vision-Language-Action Models (VLAs)

The latest paradigm: use a large pretrained vision-language model as the backbone for a robot policy, enabling **language-conditioned generalist robots**.

```
Architecture:
             Language: "pick up the red cup"
                              ↓
             Image: [camera feed of scene]
                              ↓
┌─────────────────────────────────────────────┐
│           Pretrained VLM                    │
│   (e.g., PaLM-E, RT-2, OpenVLA, π₀)       │
│   Trained on internet-scale vision+language │
└──────────────────────┬──────────────────────┘
                       ↓
             Action tokens (discretized)
                       ↓
             Robot arm joint positions / velocities
```

**RT-2 (Google, 2023):** fine-tune a pretrained VLM (PaLI) on robot demonstrations by adding action tokens to the vocabulary. The model generates actions as text tokens alongside language.

```
Input: image + "pick up the object to the right of the apple"
Output: "move_arm x=0.3 y=0.1 z=-0.2 | close_gripper | ..."
```

RT-2 achieved remarkable zero-shot generalization: it could follow instructions it had never been trained to execute, leveraging its internet-scale language understanding.

**π₀ (Physical Intelligence, 2024):** a flow-matching based VLA using a pretrained VLM (PaliGemma) + a diffusion-inspired action decoder.

```
VLM backbone (language + image understanding)
           ↓
   Action expert decoder
   (diffusion/flow matching over continuous action space)
           ↓
   6-DoF end-effector pose + gripper command
```

Flow-matching generates smooth, continuous action distributions — better than discretized token actions for fine manipulation.

---

## RL Fine-Tuning of VLAs

VLAs pretrained on demonstrations often:
- Succeed on seen tasks but fail on novel configurations
- Have systematic biases from imperfect demonstrations
- Need to improve beyond human demonstrations

**RL fine-tuning:** use PPO/SAC with the VLA as the policy, reward from task success:

```
Pretrained VLA (behavioral cloning)
         ↓
RL fine-tuning (PPO with sparse task reward)
         ↓
Policy that exceeds human demonstrations
```

Challenges:
- Large model → expensive to run many rollouts on real hardware
- Discrete action tokens → PPO for discrete actions
- Reward sparsity → need shaped rewards or careful curriculum

---

## The Full Robotics RL Stack

```
1. Collect demonstrations (teleoperation, kinesthetic teaching)
         ↓
2. Behavioral cloning / SFT on demonstrations (IL baseline)
         ↓
3. Domain randomization in simulation
         ↓
4. Policy optimization in sim (PPO/SAC + HER for manipulation)
         ↓
5. Sim-to-real transfer (domain randomization or adaptation module)
         ↓
6. Fine-tuning on real robot (offline RL on real data + safe RL)
         ↓
7. VLA: use pretrained VLM for generalization, fine-tune with RL
         ↓
Deployed robot policy
```

**State of the field (2025):**
- Quadruped locomotion: largely solved with domain randomization + RL
- Tabletop manipulation: VLAs + RL achieving impressive generalization
- Dexterous manipulation (5-finger hands): still very hard
- Long-horizon household tasks: major open challenge
