# Module 5: End-to-End Learning - Exercises

Practical exercises applying behavioral cloning, diffusion models, and reinforcement learning to robot manipulation.

---

## Exercise 1: Behavioral Cloning Grasping Policy

### Objective

Implement basic behavioral cloning to learn grasping from demonstrations.

### Scenario

You have:
- 200 labeled grasping demonstrations (image + action pairs)
- A 7-DOF arm with gripper
- Need to achieve >70% grasping success on novel objects

### Task

1. **Load and explore data**
   ```python
   demonstrations = load_grasping_dataset('grasping_200demos.pkl')

   # Analyze:
   print(f"Total demos: {len(demonstrations)}")
   print(f"Success rate: {sum(d.success for d in demonstrations) / len(demonstrations)}")

   # Visualize a few examples
   for i in range(5):
       show_image(demonstrations[i].observation)
       print(f"Action: {demonstrations[i].action}")
   ```

2. **Design network architecture**
   ```python
   policy = nn.Sequential(
       nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
       nn.ReLU(),
       nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
       nn.ReLU(),
       nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
       nn.ReLU(),
       nn.AdaptiveAvgPool2d((1, 1)),
       nn.Flatten(),
       nn.Linear(256, 128),
       nn.ReLU(),
       nn.Linear(128, 8),  # 7 joint angles + 1 gripper
   )
   ```

3. **Train with supervised learning**
   ```python
   optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

   for epoch in range(50):
       for batch in train_loader:
           images, actions = batch
           predicted = policy(images)
           loss = MSE(predicted, actions)
           loss.backward()
           optimizer.step()
   ```

4. **Evaluate on test set**
   - Measure action prediction error
   - Run on real robot with 20 test grasps
   - Measure success rate

### Success Criteria

✅ Network trains without errors
✅ Training loss decreases over epochs
✅ Test action error < 0.1 radians
✅ >70% grasping success on novel objects
✅ Can identify failure modes

### Key Insight

Behavioral cloning works well when:
- Task has single clear solution
- You have >100 quality demonstrations
- Don't need to improve beyond expert

### Solution Template

```python
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class GraspingDataset(Dataset):
    def __init__(self, demonstrations):
        self.demos = demonstrations

    def __len__(self):
        return len(self.demos)

    def __getitem__(self, idx):
        demo = self.demos[idx]
        return {
            'image': demo.observation,  # RGB
            'action': demo.action,      # [x,y,z,angle,width]
        }

# Training
policy = nn.Sequential(...)  # See architecture above
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
train_loader = DataLoader(GraspingDataset(train_demos), batch_size=16)

best_loss = float('inf')
patience = 10
no_improvement = 0

for epoch in range(100):
    for batch in train_loader:
        pred = policy(batch['image'])
        loss = F.mse_loss(pred, batch['action'])
        loss.backward()
        optimizer.step()

    val_loss = evaluate(policy, val_loader)

    if val_loss < best_loss:
        best_loss = val_loss
        no_improvement = 0
        torch.save(policy.state_dict(), 'best_policy.pth')
    else:
        no_improvement += 1
        if no_improvement > patience:
            break

# Evaluation
policy.load_state_dict(torch.load('best_policy.pth'))
success_rate = evaluate_on_robot(policy, num_trials=20)
print(f"Success rate: {success_rate:.1%}")
```

---

## Exercise 2: Diffusion Policy for Trajectory Generation

### Objective

Implement diffusion models to generate smooth robot trajectories.

### Scenario

You have:
- 300 pushing demonstrations (moving object across table)
- Images + 10-step action trajectories
- Pushing task requires smooth, coordinated motion

### Task

1. **Understand diffusion process**
   ```
   Forward (Training):
       τ₀ (clean trajectory) → noise added → τ₁ → more noise → ... → τ₁₀₀₀ (pure noise)

   Reverse (Inference):
       Pure noise → denoise step 1000→999 → ... → denoise step 1→0 → τ₀ (trajectory)
   ```

2. **Implement noise schedule**
   ```python
   class NoiseSchedule:
       def __init__(self, num_steps=1000):
           # Pre-compute variance schedule
           self.betas = torch.linspace(0.0001, 0.02, num_steps)
           self.alphas = 1 - self.betas
           self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

       def q_sample(self, x_0, t, noise):
           # x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
           return (torch.sqrt(self.alphas_cumprod[t]) * x_0 +
                   torch.sqrt(1 - self.alphas_cumprod[t]) * noise)
   ```

3. **Train denoising network**
   ```python
   policy = DiffusionPolicy()
   optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

   for epoch in range(100):
       for batch in train_loader:
           # Sample random timestep
           t = torch.randint(0, 1000, (batch_size,))

           # Add noise to trajectory
           noise = torch.randn_like(trajectory)
           x_t = noise_schedule.q_sample(trajectory, t, noise)

           # Predict noise
           noise_pred = policy(x_t, t, observation)

           # Loss: how well did we predict noise?
           loss = MSE(noise_pred, noise)
           loss.backward()
           optimizer.step()
   ```

4. **Generate trajectories at inference**
   ```python
   trajectory = policy.infer(observation, num_steps=50)
   # Output: smooth 10-step action trajectory
   ```

5. **Evaluate on test set**
   - Success rate on novel objects
   - Compare to behavioral cloning
   - Measure trajectory smoothness

### Success Criteria

✅ Diffusion loss decreases during training
✅ >80% success on held-out test objects
✅ Trajectories are smooth (no jerky motions)
✅ Better generalization than BC (60-75% on novel)
✅ Inference completes in {'<'}500ms

### Key Insight

Diffusion models handle multimodality better:
- Pushing object: multiple valid approaches (left, right, diagonal)
- BC picks average → fails
- Diffusion generates one valid approach per inference

### Solution Guide

Use the provided `diffusion_policy.py` code example:
1. Replace `SimpleEnvironment` with your grasping/pushing environment
2. Change `trajectory_length=10` to match your horizon
3. Adjust `hidden_dim=256` if GPU memory is tight
4. Train for 100-200 epochs until convergence
5. Evaluate with `num_steps=50` for inference (balance quality vs speed)

---

## Exercise 3: Reinforcement Learning Policy

### Objective

Implement PPO to learn robot manipulation through trial-and-error.

### Scenario

You have:
- Simulation environment (Gazebo or MuJoCo)
- Dense reward signal (distance to goal)
- Need to learn complex multi-step manipulation

### Task

1. **Define environment and reward**
   ```python
   class ManipulationEnv:
       def reset(self):
           return observation

       def step(self, action):
           # Execute action
           next_obs = move_robot(action)

           # Reward: negative distance to goal
           distance = ||next_obs.object_pos - goal_pos||
           reward = -distance

           return next_obs, reward, done
   ```

2. **Implement policy with actor-critic**
   ```python
   policy = PPOPolicy(
       obs_dim=10,          # Observation dimension
       action_dim=4,        # Action dimension
       hidden_dim=256,
       continuous=True,
   )
   ```

3. **Collect episodes and train**
   ```python
   for episode in range(10000):
       # Collect trajectory
       obs = env.reset()
       for step in range(100):
           action = policy.select_action(obs)
           next_obs, reward, done = env.step(action)

           # Store in buffer
           policy.store_transition(Transition(
               state=obs,
               action=action,
               reward=reward,
               done=done,
           ))

           obs = next_obs
           if done:
               break

       # Train every 10 episodes
       if (episode + 1) % 10 == 0:
           policy.train_step(batch_size=64, epochs=3)

           # Evaluate
           test_reward = run_test_episode(env, policy, deterministic=True)
           print(f"Episode {episode}, Test Reward: {test_reward:.3f}")
   ```

4. **Evaluate on test distribution**
   ```python
   test_rewards = []
   for _ in range(50):
       obs = env.reset()
       reward = 0
       for _ in range(100):
           action = policy.select_action(obs, deterministic=True)
           obs, r, done = env.step(action)
           reward += r
           if done:
               break
       test_rewards.append(reward)

   print(f"Test return: {np.mean(test_rewards):.3f} ± {np.std(test_rewards):.3f}")
   ```

5. **Analyze learned behavior**
   - Visualize trajectories
   - Identify exploration strategies
   - Compare to BC baseline

### Success Criteria

✅ Episode rewards increase over time (learning signal)
✅ Convergence within 5000 episodes
✅ >85% success on test distribution
✅ Better performance than BC baseline (if applicable)
✅ Stable learning (low variance in test returns)

### Key Insight

RL beats demonstrations when:
- Task has multiple valid solutions (explore to find good ones)
- Demonstrations are suboptimal (learn to improve)
- Complex long-horizon tasks (exploration helps)

### Solution Guide

Use provided `rl_policy.py`:
1. Replace `SimpleEnvironment` with your robot environment
2. Adjust reward function for your task
3. Tune hyperparameters:
   - `learning_rate`: Start with 1e-4, reduce if diverging
   - `clip_ratio`: Keep at 0.2 (PPO standard)
   - `entropy_coef`: Increase if under-exploring (0.01 → 0.05)
4. Monitor loss values (should be stable, not spiking)
5. Early stop if test return plateaus for 500 episodes

---

## Challenge Exercises (Optional)

### Challenge 1: Multi-Task Learning with BC

**Difficulty**: Medium (2-3 hours)

Extend behavioral cloning to multiple tasks:
- Collect 100 demos each for: grasping, pushing, insertion
- Single policy predicts actions for all three tasks
- Condition on task ID: policy(image, task_id) → action

**Success**: 70%+ success per task, better generalization to new tasks

---

### Challenge 2: DAgger for Distribution Correction

**Difficulty**: Medium (3-4 hours)

Implement DAgger to fix BC errors:
1. Train initial BC policy
2. Deploy on robot, collect failures
3. Expert corrects: policy makes mistake → expert shows right action
4. Add corrections to training data
5. Retrain
6. Repeat 3-5 times

**Success**: Improve from 70% → 85%+ via corrections

---

### Challenge 3: Diffusion + RL Hybrid

**Difficulty**: Hard (5-7 hours)

Combine diffusion policy with RL:
1. Train diffusion policy on demonstrations (85% success)
2. Fine-tune with RL for 1-2 days
3. Compare to: BC alone, Diffusion alone, RL alone

**Success**: Hybrid achieves 92%+ (beats all individual methods)

---

## Comparison Exercise

### Task: Grasping Comparison

Same task, three methods:

| Method | Time | Data | Success | Generalization |
|--------|------|------|---------|-----------------|
| BC | 4 hours | 200 demos | 73% | 45% (novel) |
| Diffusion | 2 days | 200 demos | 86% | 62% (novel) |
| RL | 2 weeks | 0 demos | 94% | 78% (novel) |
| BC+RL | 1 week | 200 demos | 91% | 72% (novel) |

**Report**: Measure all four on same grasping task, compare results

---

## Summary

You now understand:

✅ **Behavioral Cloning**: Why copying demonstrations works and fails
✅ **Diffusion Models**: Iterative trajectory generation handles multimodality
✅ **Reinforcement Learning**: Trial-and-error learning beats demonstrations
✅ **End-to-End Learning**: Direct mapping from pixels to motor commands
✅ **Method Selection**: Choose BC/Diffusion/RL based on task and constraints
✅ **Practical Deployment**: Trade-offs between speed, quality, and data efficiency

---

## Further Reading

- **Behavioral Cloning**: Imitation Learning from Observations (Sermanet et al., 2018)
- **Diffusion Policy**: Diffusion Models as Generative Priors (Chi et al., 2023)
- **PPO**: Proximal Policy Optimization Algorithms (Schulman et al., 2017)
- **RT-2**: Comparing Different Approaches to Open-Vocab Robot Manipulation (Brohan et al., 2023)

---

**Congratulations!** You've completed Module 5. You now understand the complete pipeline from raw sensory data to robot motor commands.

**Next Module**: [Module 6: Scaling & Production Systems](../scaling-systems/intro.md) - How to train policies on thousands of tasks and deploy at scale.

