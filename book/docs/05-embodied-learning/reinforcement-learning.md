# Reinforcement Learning for Embodied AI

**Core idea**: Trial-and-error learning. Robot executes action, receives reward signal, learns to maximize cumulative reward.

---

## Why RL Instead of BC/Diffusion?

### The Limitation of Demonstrations

Behavioral cloning and diffusion learn from human-generated trajectories. But:

1. **Humans are bounded** - Can't demonstrate superhuman performance
2. **Suboptimal demonstrations** - Humans make mistakes
3. **Narrow policy** - Only learns the strategy shown in demos

### The RL Advantage

```
BC/Diffusion:  Copy expert behavior exactly
               Success rate = Expert success rate (70-90%)

RL:            Improve beyond expert, explore better strategies
               Success rate > Expert success rate (90-98%)
```

**Real example: Robotic Manipulation**

```
Expert grasping success rate: 88%
BC learns to copy: 85% (close to expert)
RL trained with reward "grasp success": 96% (beats expert)

Why? RL finds grips human didn't try. More finger angles, different approach paths.
```

---

## Core RL Concepts

### The MDP (Markov Decision Process)

Robot learning loop:

```
State s_t        (observation: image, joint angles, forces)
     ↓
Action a_t       (gripper command: x, y, z, width)
     ↓
Reward r_t       (feedback: grasp succeeded +1, failed 0, etc.)
     ↓
Next state s_{t+1} (gripper closed on object, moved to new location, etc.)
```

**Goal**: Learn policy π(a_t | s_t) that maximizes cumulative reward:

```
R = Σ γ^t · r_t

where γ = discount factor (0.99 = weight recent rewards more)
```

### Policy Gradient: The Key Insight

**Goal**: Make actions that led to high reward more likely

```python
# If grasp succeeded (reward = 1):
    increase probability of that action
    increase probability of that observation → action mapping

# If grasp failed (reward = 0):
    decrease probability of that action
```

**Mathematically**:
```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) · R]

Gradient in direction that increases log probability × reward
If reward is high: stronger gradient
If reward is low: weak gradient
```

---

## Policy Gradient Methods (PG)

### Simple Algorithm: REINFORCE

```python
# Collect trajectory
trajectory = []
state = env.reset()
for t in range(T):
    action = policy.sample(state)  # Sample from network
    next_state, reward = env.step(action)
    trajectory.append((state, action, reward))
    state = next_state

# Compute return (cumulative discounted reward)
returns = []
cumulative = 0
for reward in reversed(trajectory):
    cumulative = reward + gamma * cumulative
    returns.insert(0, cumulative)

# Update policy
for (state, action, return_t) in trajectory:
    loss = -log(π(action|state)) * return_t  # Policy gradient
    loss.backward()
    optimizer.step()
```

**Why it works**:
- High return → large loss → strong gradient to increase action probability
- Low return → small loss → weak gradient
- Over many trajectories → naturally converges to better policy

### Practical Example: Learning to Reach

```
Task: Move robot arm end-effector to target position
Observation: Image of target
Action: Joint velocity [θ̇₁, ..., θ̇₇]
Reward: -distance_to_target (penalize errors)

Episode 1:
  Random exploration: distance = 0.5m, return = -0.5
  Policy slightly learns to move toward target

Episode 10:
  Better exploration: distance = 0.2m, return = -0.2
  Policy stronger learning signal

Episode 100:
  Good strategy learned: distance = 0.05m, return = -0.05
  Convergence
```

---

## Advanced: PPO (Proximal Policy Optimization)

### Problem with REINFORCE

Basic policy gradient has high variance:

```
Same trajectory executed 10 times:
- Run 1: Return = 10.2 (variance from stochasticity)
- Run 2: Return = 9.8
- Run 3: Return = 10.1
Average variance ≈ 0.15

Policy updates are noisy, inefficient
```

### PPO Solution

Use two networks:

```
Actor (π):   State → Action distribution
             Learns what action to take

Critic (V):  State → Expected return
             Learns to predict trajectory quality
```

**Idea**: Use critic to reduce variance

```python
# Compute advantage (how much better than expected)
advantage = actual_return - critic_prediction(state)

# Update policy only when advantage > 0
if advantage > 0:
    increase action probability
else:
    decrease action probability (trust critic, not this one trajectory)
```

**Result**:
- Much lower variance (critic removes common component)
- Faster convergence (data efficiency)
- 50% less data needed vs REINFORCE

---

## Actor-Critic Architecture

### Network Structure

```
Input: observation (image, proprioception)

Shared encoder:
  CNN: image → 256D features
  FC: properties → 64D features
  Concatenate: 256 + 64 = 320D

Actor head (policy):
  MLP: 320 → 256 → 128 → action_dim
  Output: action mean and std dev

Critic head (value):
  MLP: 320 → 256 → 128 → 1
  Output: predicted return (scalar)
```

### Training Loop (PPO)

```python
for epoch in range(100):
    # Collect trajectories
    trajectories = []
    for _ in range(num_workers=4):  # Parallel collection
        state = env.reset()
        trajectory = []
        for t in range(T):
            action = actor.sample(state)
            next_state, reward = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
        trajectories.extend(trajectory)

    # Compute advantages and targets
    for (state, action, reward) in trajectories:
        returns[state] = reward + gamma * returns[next_state]
        advantage[state] = returns[state] - critic(state)

    # PPO update (with clipping to prevent overstep)
    for epoch_step in range(K):  # K passes through data
        old_log_probs = old_actor.log_prob(actions)
        new_log_probs = actor.log_prob(actions)

        ratio = exp(new_log_probs - old_log_probs)

        # Clipped objective - don't update too much
        clipped = clip(ratio, 1-ε, 1+ε)
        actor_loss = -min(ratio, clipped) * advantage

        critic_loss = (returns - critic(state))²

        total_loss = actor_loss + 0.5 * critic_loss
        total_loss.backward()
        optimizer.step()
```

---

## Reward Engineering: The Hard Part

Designing the reward signal is **critical and difficult**.

### Example: Grasping Task

```
Option 1: Sparse reward
    reward = 1 if grasp successful, 0 otherwise
    Problem: No learning signal during most of episode
    Takes 10K+ trials to find success by random exploration

Option 2: Dense reward (better)
    reward = 1 - distance_to_target  (penalize distance)
           + 1 - angle_error         (penalize wrong angle)
           - gripper_collision        (penalize hitting table)
    Problem: Multiple reward terms, hard to balance
    Example: If negative collision term too strong, robot never grasps
             If distance term too weak, robot ignores target

Option 3: Reward shaping (best)
    reward = 1 - distance_to_target
    + bonus_if(gripper_closed AND object_in_gripper)
    Problem: Still requires manual engineering
    Time: 1-2 weeks to get right
```

### Real Example: OpenAI Dactyl Hand

```
Task: Manipulate cube to target position using dexterous hand (16 DOF)

Reward crafted over weeks:
    r = 0.3 · (1 - ||pos_error||²)      [reach target]
      + 0.4 · (1 - ||orientation||²)   [match orientation]
      + 0.2 · contact_success           [maintain contact]
      + 0.1 · smoothness_penalty        [smooth motion]

Each coefficient tuned based on:
    - Success rate on different object sizes
    - Convergence speed
    - Movement quality (no jerky motions)

Final tuning took ~2 weeks of engineering
```

---

## Simulation vs Real World

### The Reality: Sim-to-Real Gap is Real

Most robot RL training happens in simulation (Gazebo, MuJoCo, Isaac Sim):
- 100-1000× faster than real robots
- Infinite data (no robot wear)
- Safe to experiment (no crashes)

But learned policies often fail on real robots.

### Solution 1: Domain Randomization

Train with random variations:
```python
def randomize_simulation():
    # Physics parameters
    gravity = random(9.6, 9.9)  # Real gravity ~9.81
    friction = random(0.3, 1.2)  # Real friction varies
    mass = original_mass * random(0.8, 1.2)  # Real objects vary

    # Visual parameters
    object_colors = random_colors()
    lighting = random_lighting()
    camera_noise = add_gaussian_noise(std=0.01)

    return randomized_env

# Train with diverse environments
for episode in range(10000):
    env = randomize_simulation()
    trajectory = collect_episode(env)
    update_policy(trajectory)
```

**Result**: Policy sees thousands of different environments, learns robust strategy that works on new real environments

**Success rate**:
- No randomization: 30% on real robot
- With randomization: 76% on real robot

### Solution 2: Real Robot Training

**Expensive but direct**:
```
- Start with RL in simulation
- Fine-tune on real robot (1-2 hours)
- Policy transfers well
```

**Trade-off**:
- Cost: $100/hour of robot time
- Benefit: Gets exactly the real robot dynamics

---

## RL vs BC vs Diffusion: When to Use What

| Criterion | BC | Diffusion | RL |
|-----------|----|-----------|----|
| **Demo quality requirement** | High (needs good demos) | High | None (learns from scratch) |
| **Sample efficiency** | 100 demos | 200 demos | 10K-100K trials |
| **Training time** | Hours | 4-8 hours | 1-4 weeks |
| **Success rate ceiling** | 70-80% | 85-90% | 90-98% |
| **Complexity to implement** | Easy | Medium | Hard (reward engineering) |
| **Generalization** | Poor | Good | Excellent |
| **Exploration capability** | None | Limited | Extensive |
| **Superhuman performance** | Impossible | Unlikely | Possible |

---

## Practical RL Checklist

### Before Starting RL

- [ ] Clear reward signal (can it be tested?)
- [ ] Simulation environment (speed > 10 FPS?)
- [ ] Baseline BC policy (what's expert level?)
- [ ] ~50 demonstrations for reward signal validation
- [ ] GPU access (RL = compute intensive)

### During Training

- [ ] Monitor success rate over time (plateaus = sign of problem)
- [ ] Visualize learned behaviors (is policy doing what expected?)
- [ ] Early stopping (stop if not improving after 1000 episodes)
- [ ] Reward logging (understand reward signal composition)

### Debugging Failed RL

```
Problem 1: Reward always zero (no signal)
Solution: Check reward computation
         Add intermediate rewards for progress

Problem 2: Success rate plateaus at 30-40%
Solution: Reward may be too sparse
         Add shaped rewards for intermediate progress
         Or use demonstrations to warm-start policy

Problem 3: Success rate varies wildly (50-90%)
Solution: High variance, environment stochasticity
         Increase exploration time
         Reduce environment randomization initially
```

---

## Real-World Robotics Examples

### Google Robotics (Gato)

**Setup**:
- 600 robots
- 4M manipulations collected
- RL + behavioral cloning (hybrid)

**Results**:
- 90%+ success on 700+ tasks
- Transfer to new robots without retraining

### DeepMind Robotics (MuZero)

**Setup**:
- Model-based RL (learns world model + policy)
- Simulation training only

**Results**:
- Zero real robot data
- 85% success on real robot reaching task
- Model-based approach is more sample-efficient

### OpenAI Dactyl

**Setup**:
- Dexterous hand (16 DOF) in simulation
- Domain randomization (1000× variations)
- PPO training for 1 month (GPU cluster)

**Results**:
- First dexterous manipulation on real hand
- 76% cube reorientation on real robot
- Zero real robot training (pure sim-to-real)

---

## Key Takeaways

✅ **RL beats demonstrations** - Can achieve superhuman performance
✅ **PPO is practical** - State-of-the-art, easy to implement
✅ **Reward engineering is hard** - 50% of RL effort is reward design
✅ **Simulation is key** - Domain randomization for sim-to-real transfer
✅ **Actor-critic works** - Critic reduces variance, faster convergence
✅ **Sample efficiency matters** - RL needs 10-100K trials

---

## Next Steps

1. **Understand the reward signal** - What does "success" mean for your task?
2. **Set up simulation** - Gazebo, MuJoCo, or Isaac Sim
3. **Implement basic PG** - Start with REINFORCE, then move to PPO
4. **Add domain randomization** - Prepare for sim-to-real transfer
5. **Benchmark against BC** - Compare methods on same task

---

## Further Reading

- **Policy Gradient Methods** (Sutton & Barto, 2018): Foundational RL theory
- **PPO** (Schulman et al., 2017): State-of-the-art policy gradient method
- **Dactyl Hand** (OpenAI, 2018): Domain randomization for sim-to-real
- **RT-1** (Brohan et al., 2022): RL + BC hybrid for manipulation
- **Gato** (DeepMind, 2022): Multi-task RL at massive scale

---

**Next Section:** [End-to-End Learning Architectures →](end-to-end-learning.md)

