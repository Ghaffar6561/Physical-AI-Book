# Imitation Learning: Learning from Demonstrations

Imitation learning is the simplest approach: **watch demonstrations, predict actions, that's it.** No reward signals. No trial-and-error. Just supervised learning.

---

## The Core Idea

```
Step 1: Collect demonstrations
       (Expert executes task, we record: image, action taken)

Step 2: Supervised learning
       image → network → predicted action
       Loss = (predicted_action - expert_action)²

Step 3: Deployment
       new image → network → action
```

**Why this works**: The network learns a compressed representation of expert behavior.

**Why this fails**: The network only sees good trajectories. When it makes a small mistake, it enters a state the expert never demonstrated, and error compounds.

---

## Behavioral Cloning (BC): The Simple Version

### Basic Algorithm

```python
# Collect demonstrations: D = [(image_0, action_0), ..., (image_N, action_N)]

# Train a policy π_θ(a | image) to predict actions given images
for epoch in range(100):
    for batch in dataloader:
        images, expert_actions = batch
        predicted_actions = policy(images)
        loss = MSE(predicted_actions, expert_actions)
        loss.backward()
        optimizer.step()

# Deploy: given new image, use policy to predict action
new_image = camera.capture()
action = policy(new_image)
robot.execute(action)
```

### What Gets Learned

The network learns a **conditional probability distribution**: P(action | image)

In practice, this means learning:
- **Where to look** (what features matter)
- **How to map** visual features to action parameters
- **Implicit constraints** (what actions are physically possible)

### Real Example: Grasping

```
Training data: 1000 successful grasps
- Image: top-down RGB of object on table
- Action: [gripper_x, gripper_y, gripper_angle, gripper_width]

Network learns:
- Red objects → grasp from north
- Blue objects → grasp from east
- Spheres → grasp from top
- etc.

Learned distribution P(a | image) captures: "most grasps work from the side"
```

---

## The Compounding Error Problem

This is the core limitation of behavioral cloning.

### What Happens in Practice

```
Step 1: Policy makes small prediction error (5% wrong angle)
Step 2: Robot ends up in slightly wrong state
Step 3: This state was NEVER in training data
Step 4: Policy is confused, makes bigger error
Step 5: Robot crashes or fails
```

### Mathematical View

**At distribution**:
```
P_expert(s) = state distribution under expert behavior
Network trained on expert states: s ~ P_expert
```

**At test time**:
```
P_policy(s) ≠ P_expert(s)  ← Policy makes mistakes, visits different states
Network has never seen these states → predictions are random
Prediction error → worse state → exponential error accumulation
```

**Result**:
- Expert success rate: 95%
- BC success rate: 40-60% (even though network classifies correctly 90% of the time)

### Concrete Example: Pushing

```
Expert trajectory:
Image1 → Action(push_right) → Image2
Image2 → Action(push_forward) → Image3
Image3 → Action(stop) → Success

BC trajectory:
Image1 → Action(push_right + 0.1 error) → Image2'
                                           ↑ NOT in training data
Image2' → ??? (network confused)
         → Action(push_left) ← makes error worse
Image3' → Action(stop)
         → FAILURE
```

**Summary**: Behavioral cloning works well for 70-80% of cases but fails catastrophically when it makes mistakes.

---

## DAgger: Learning from Intervention

**Key insight**: We can fix BC by teaching it what to do when it makes mistakes.

### The Algorithm

```python
# Start with behavioral cloning
policy = train_bc_policy(demonstrations)

# Iteratively improve
for iteration in range(10):
    # Run policy in environment
    trajectory = []
    state = env.reset()

    for step in range(100):
        # Execute policy
        action = policy(state)

        # Check if policy is making a mistake
        if is_confident(policy, state) and action_looks_bad(action):
            # Ask expert what to do (intervention)
            expert_action = expert.suggest_action(state)
            trajectory.append((state, expert_action))
        else:
            # Trust policy
            next_state = env.step(action)
            state = next_state

    # Add new trajectory to dataset
    demonstrations.add(trajectory)

    # Retrain on larger dataset
    policy = train_bc_policy(demonstrations)
```

### Why DAgger Works

**It breaks the distribution mismatch problem**:

**BC**: Train only on expert states
```
Policy trained on: {expert states}
Policy tested on: {policy states}  ← Different! Error accumulates.
```

**DAgger**: Train on both expert AND policy states
```
Policy trained on: {expert states} + {mistakes policy made}
Policy tested on: {policy states}  ← Same! No distribution mismatch.
```

### Real-World Results

**Study**: Learning to drive a car (Bojarski et al., 2016)

| Method | Success Rate |
|--------|-------------|
| BC only | 45% |
| BC + 500 DAgger interventions | 68% |
| BC + 1500 DAgger interventions | 82% |
| Expert driver | 99% |

**Key observation**: With ~15 corrective interventions per 100 driving steps, achieves 82% success

---

## Multi-Task Behavioral Cloning

### Single-Task vs Multi-Task

**Single-task BC**:
```
Grasping only:
- Train 1 network per task
- 88% success
- Doesn't generalize to pushing
```

**Multi-task BC**:
```
Grasping + pushing + insertion:
- 1 network for all tasks
- Condition on task ID: policy(image, task_id) → action
- 72% success on all tasks (slight drop but huge generalization)
```

### Architecture for Multi-Task Learning

```
Input: image (224×224×3)
       task_embedding (8D vector for "grasp"/"push"/"insert")

Shared encoder: image → 256D features
Task encoder: task_id → 8D embedding

Fusion: concatenate [features, task_embedding]
         → MLP(256+8 → 128 → 64)

Output: action (4D: x, y, angle, width)

Loss per task: weighted by success rate
   total_loss = w_grasp * loss_grasp + w_push * loss_push + ...
```

### Real-World Example: RT-1

**Google's RT-1 robot**:
- Trained on 130K episodes across 700+ tasks
- Same network for: grasping, pushing, opening drawers, sorting
- 76% success on seen tasks, 26% on novel tasks
- Multi-task learning helped: single-task BC got only 60% even on seen tasks

**Why multi-task helps**:
1. Shared visual understanding (edges, colors, shapes help all tasks)
2. Larger dataset (700 tasks = much more data)
3. Implicit regularization (can't overfit when predicting diverse actions)

---

## Key Limitations of BC/DAgger

| Limitation | Why It Matters | Mitigation |
|-----------|---------------|-----------|
| **Distribution mismatch** | Errors compound | DAgger adds expert corrections |
| **Off-policy data** | Can't learn from failure trajectories | Could use RL, but slower |
| **Multimodal actions** | "Pick cup from left OR right" - BC picks one | Diffusion models handle better |
| **Requires good demos** | Bad demonstrations → bad policy | Can't be worse than demonstrator |
| **Task-specific** | Learned grasping doesn't help with pushing | Multi-task helps, still limited |
| **No exploration** | Can't improve beyond demonstrator | RL can explore, BC can't |

---

## Success Criteria for BC

Use behavioral cloning when:

✅ You have 100+ clean demonstrations
✅ Task has single clear solution (open door, push button)
✅ Don't need superhuman performance
✅ Deployment is quick (hours, not weeks)
✅ Can afford to pay expert for DAgger interventions

❌ Don't use when:
❌ Task has multiple valid solutions (many ways to grasp)
❌ Need superhuman performance
❌ Complex task requiring exploration
❌ Distribution shift expected (new objects, new robot)

---

## Comparison Table

| Approach | Data Cost | Time to Deploy | Success Rate | Generalization |
|----------|-----------|----------------|-------------|-----------------|
| **BC** | 100-500 demos | Hours | 70-80% | Poor (30-50% on novel) |
| **BC + DAgger** | 100 demos + 500 interventions | Days | 75-85% | Fair (40-60% on novel) |
| **Multi-task BC** | 10K+ episodes across many tasks | Weeks | 70-80% | Better (50-70% on novel) |
| **Diffusion** | 200-500 demos | Days | 80-90% | Better (60-70% on novel) |
| **RL** | ~10K trials | 1-4 weeks | 85-95% | Best (70-80% on novel) |

---

## Real-World Deployment Considerations

### Data Collection

**The bottleneck**: Getting demonstrations

```
Cost breakdown:
- Robot hardware: $50K-$500K (one-time)
- Expert time: $50-150/hour (20-50 hours for 500 demos)
- Data labeling: 30-60 seconds per demo
- Total: $1K-$10K per task
```

**In practice**:
- Self-play: Use previous policies to generate demos (biased but cheap)
- Human kinesthetic teaching: Move robot arm by hand (slow but natural)
- Remote operation: Expert controls robot via interface (fast, expensive)

### Data Quality

```
Expert success rate: 95%
Training on 100 demos: 2-3 failures included (bad data)
BC learns from failures: Policy becomes less reliable

Mitigation:
1. Filter out low-confidence trajectories
2. Require multiple successful demos per state
3. Use automatic quality scoring (replay trajectory, see if successful)
```

### Network Architecture Choices

**Simple CNN** (recommended for start):
```
Input: 224×224 RGB image
Conv layers: 3×(64 channels, 3×3 kernel, ReLU)
Average pool: 7×7
FC layers: 256 → 128 → action_dim
Success rate: 70-80%
Inference: 50ms
```

**Vision Transformer (ViT)**:
```
Input: 224×224 RGB image (split into patches)
Transformer blocks: 12 layers, 12 heads
Output: Classification → regression to action
Success rate: 75-85% (better generalization)
Inference: 200ms (slower, not real-time)
```

**Recommendation**: Start with simple CNN for fast iteration, switch to ViT if generalization is critical.

---

## Debugging Failed BC Policies

When your behavioral cloning policy is failing:

### Check 1: Dataset Quality
```python
# Visualize what network sees
for trajectory in demonstrations:
    images, actions = trajectory
    predicted = policy(images)
    error = |predicted - actions|

    if error > threshold:
        print(f"Bad demo: {error:.3f}")  # Learn from data quality issues
```

### Check 2: Action Space
```python
# Are actions continuous or discrete?
# Continuous: predict [x, y, angle, width] directly
# Discrete: predict "grasp_left", "grasp_right", etc.
# Mismatch = poor learning
```

### Check 3: Network Capacity
```
Test 1: Can network memorize training data?
    Train to 0% error on same episodes
    If not: network too small, increase hidden layers

Test 2: Does it overfit training data?
    Train accuracy: 95%, Test accuracy: 40%
    If yes: add dropout, reduce layer size, or more data
```

### Check 3: Action Prediction vs Execution
```
network predicts: [0.3, 0.2, 45°, 0.08]
robot receives:   [0.3, 0.2, 45°, 0.08]
robot executes:   [0.29, 0.21, 46°, 0.078]  ← Actuator noise

This 5-10% execution error is NORMAL. Plan for it.
```

---

## Key Takeaways

✅ **Behavioral cloning works** for single-solution tasks with good demonstrations
✅ **Distribution mismatch** is the core problem (errors compound)
✅ **DAgger fixes this** by collecting corrections to states policy visits
✅ **Multi-task learning** helps generalization, especially with large datasets
✅ **But BC has limits** - multimodal tasks need diffusion/RL

---

## Next Steps

1. **Understand the failure mode** - Distribution shift, not bad demonstrations
2. **Try DAgger** if you have ~500 interventions available
3. **Consider multi-task** if you have demonstrations across similar tasks
4. **For complex tasks** - move to diffusion models (next section) or RL

---

## Further Reading

- **ALVINN** (Pomerleau, 1989): Original behavioral cloning for autonomous driving
- **DAgger** (Ross, Gordon, Barto, 2011): Dataset aggregation fixing distribution mismatch
- **Behavioral Cloning from Observation** (Torabi et al., 2018): Learning without action labels
- **RT-1** (Brohan et al., 2022): Multi-task BC at scale (700+ tasks, 130K episodes)

---

**Next Section:** [Diffusion Models for Robot Control →](diffusion-for-robotics.md)

