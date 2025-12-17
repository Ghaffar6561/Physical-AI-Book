# End-to-End Learning: Architectures & Strategies

**Core principle**: Map directly from raw observations (images, proprioception) to motor commands. No intermediate modules.

---

## Why End-to-End?

### Modular vs End-to-End

**Modular Approach** (from Module 4):
```
Image → [Object detection] → [Target estimation] → [IK solver] → Joint angles
         (hand-crafted)        (hand-crafted)      (hand-crafted)

Problem: Each module has errors that accumulate
Example: Detection 95% × target estimation 90% × IK 98% = 84% total
```

**End-to-End Learning**:
```
Image → [Single neural network] → Joint angles
        Learns entire mapping automatically

Advantage: Network learns to work around errors in earlier stages
          Can't accumulate errors in intermediate representations
```

### Real Example: NVIDIA's Autonomous Driving

**Traditional modular approach**:
```
Camera → Lane detection → Steering angle
         (hand-crafted)
- Hard to handle edge cases
- Lane detection fails in shadows, rain
- Manual tuning required per environment
```

**End-to-end approach** (Bojarski et al., 2016):
```
Camera → CNN → Steering angle
        Learns to ignore rain drops, shadows, etc.
        Single model, 10K hours of driving data
- 90% success on real roads
- Automatic adaptation to conditions
```

---

## Architecture Families

### 1. Convolutional Neural Networks (CNN)

**Best for**: Visual tasks with clear spatial patterns (grasping, reaching)

```
Input: Image (224×224×3)
        ↓
Conv Layer 1:  64 filters, 3×3 kernel → (224, 224, 64)
               ReLU activation
Conv Layer 2:  128 filters, 3×3 kernel → (112, 112, 128)
               Max pool
Conv Layer 3:  256 filters, 3×3 kernel → (56, 56, 256)
               Max pool
Flatten:       → (50176,)
               ↓
FC Layer 1:    → (512,) + ReLU + Dropout(0.3)
FC Layer 2:    → (128,) + ReLU
Output:        → (7,) joint angles + (1,) gripper width

Properties:
- Spatial inductive bias (nearby pixels are related)
- Parameter efficient (~2M parameters)
- Fast inference (50-100ms)
- Good for learned features
```

### 2. Vision Transformers (ViT)

**Best for**: Complex reasoning tasks (rearrangement, long-horizon)

```
Input: Image split into patches
       16×16 patches = 14×14 = 196 patches

Patch embedding: Each patch → 768D

Transformer blocks (12 layers):
  Self-attention: Learn which patches are important
  Feed-forward: Process patch information

Classification head: [CLS] token → 8D (output actions)

Properties:
- Global attention (can see entire image at once)
- Better generalization to new objects
- Slower inference (200-500ms)
- More parameters (~100M for large models)
- Needs more training data (10K+ demonstrations)
```

### 3. Hybrid: CNN + RNN

**Best for**: Tasks with temporal dependencies (pushing, insertion)

```
Input sequence: Image_t, Image_{t-1}, Image_{t-2} (last 3 frames)
                    ↓
CNN encoder:    Extract spatial features from each image
                Conv3D with temporal kernel → (64, 3, 56, 56)
                    ↓
LSTM:           Process sequence of features
                Input: 64 features per frame × 3 frames
                Hidden: 256 neurons
                Output: Cell state captures motion
                    ↓
FC layers:      Decode motion understanding to action
                → 8D action

Why temporal?:
- Pushing: Direction matters (push consistent direction)
- Insertion: Alignment matters (subtle adjustments needed)
- Without RNN: Network only sees single frame, can't learn timing
```

---

## Training Strategies

### Strategy 1: Behavior Cloning (BC)

```
# Data collection
Collect demonstrations: (image, action) pairs
Expert success rate: 88%

# Training
for epoch in range(50):
    loss = MSE(network(images), expert_actions)
    loss.backward()
    optimizer.step()

# Results
Training accuracy: 92% (network matches expert)
Test accuracy: 45% (on novel objects)
↑ High variance - distribution mismatch
```

**When to use**: Quick deployment, {'<'}100 demonstrations available

**Results**: 60-75% success on in-distribution tasks

---

### Strategy 2: Behavioral Cloning + Fine-tuning with RL

**Best of both worlds**:

```
Step 1: Warm-start with BC (2-4 hours)
        network = train_bc(demonstrations)
        Result: Network knows approximate mapping

Step 2: Fine-tune with RL (2-4 days)
        for episode in range(10000):
            trajectory = collect_episode(network)
            rl_loss = -log(network(state)) * advantage
            bc_loss = 0.1 * MSE(network(state), expert_action)
                      ↑ Keep some BC signal to avoid forgetting

            total_loss = rl_loss + bc_loss
            total_loss.backward()

Result: Network improves beyond expert while staying stable
```

**Advantage**:
- Starts at 75% (BC baseline)
- Improves to 90% (RL fine-tuning)
- Doesn't diverge to random behavior
- Combines data efficiency of BC + performance of RL

**Real example: RT-2**
```
Pre-training: BC on 130K demonstrations → 76% success
Fine-tuning: RL on 10K additional trials → 91% success
Total: Expert-level performance on diverse tasks
```

---

### Strategy 3: Multi-Task Learning

**Leverage data across multiple tasks**:

```
# Data:
5000 grasping demonstrations
3000 pushing demonstrations
2000 insertion demonstrations

# Architecture:
shared_encoder: image → 256D features (shared for all tasks)

task_embedding: task_id → 32D (unique per task)

fusion: [shared_features, task_embedding] → 288D
        ↓
shared_head: → 128D

task-specific heads:
  grasp_head: 128D → 4D (target x,y, width, angle)
  push_head:  128D → 2D (push direction, force)
  insert_head: 128D → 3D (insertion depth, angle, speed)

# Training:
Loss = w_grasp * L_grasp + w_push * L_push + w_insert * L_insert

where w = task loss / total loss (tasks with higher loss get more weight)
```

**Results**:
- Single-task BC: 75% per task
- Multi-task BC: 68% per task (slight drop)
- On NEW task (not in training): 45% for single-task, 52% for multi-task

**Key benefit**: Transfer learning - pre-train on 10 tasks, fine-tune on new task with 100 demos

---

## Key Architectural Decisions

### Decision 1: Input Representation

| Input Type | Pros | Cons | Best For |
|-----------|------|------|----------|
| **RGB only** | Simple, works | Color is fragile, needs more data | Grasping with clear objects |
| **RGB + Depth** | Spatial info, robust | Requires depth camera | Precise manipulation |
| **RGB + Proprioception** | Full state, no ambiguity | Proprioception can be noisy | Any task with arm |
| **Point cloud** | 3D, rotation invariant | Slow, complex | Complex environments |

**Recommendation**: RGB + proprioception (joint angles, gripper state)

### Decision 2: Output Representation

| Output Type | Pros | Cons | Best For |
|------------|------|------|----------|
| **Joint angles** | Direct control | Hard to generalize across robots | Single specific robot |
| **Task space** (x,y,z,gripper) | Generalize to new robots | Requires IK solver | Multiple robot morphologies |
| **Action tokens** (discrete) | Stable, interpretable | Limited flexibility | Pre-defined primitives |
| **Action parameters** (continuous) | Smooth, flexible | Can be unstable | Most cases |

**Recommendation**: Task-space with IK (x,y,z,gripper_width,grasp_force)

### Decision 3: Network Size

```
Task complexity vs data available:

Simple tasks (grasp):
- Grasping: one object, single solution
- Network: 2-4 CNN layers, 2M parameters
- Data needed: 100 demonstrations
- Success: 85%

Medium tasks (rearrangement):
- Multi-object, reasoning needed
- Network: 6-8 CNN layers, 10-50M parameters
- Data needed: 1000 demonstrations
- Success: 75%

Complex tasks (long-horizon):
- Multi-step, planning needed
- Network: Vision Transformer, 100M+ parameters
- Data needed: 10K+ demonstrations
- Success: 60-70% (multi-step is hard)

Rule of thumb: network_params = num_demonstrations / 10
(100 demos → 10M parameters, 1000 demos → 100M parameters)
```

---

## Training Best Practices

### 1. Data Augmentation

Real data is expensive. Synthetic augmentation helps:

```python
import torchvision.transforms as T

augmentation = T.Compose([
    T.RandomRotation(degrees=15),      # Robot can see object at different angles
    T.RandomAffine(degrees=0,          # Translation
                   translate=(0.1, 0.1)),
    T.ColorJitter(brightness=0.2,      # Lighting variations
                  contrast=0.2,
                  saturation=0.2),
    T.GaussianBlur(kernel_size=3),     # Camera noise
    T.RandomErasing(p=0.1),            # Occlusion
])

# Apply to training data only
for image in train_images:
    image_aug = augmentation(image)
    # Network trained on both original and augmented
```

**Result**:
- No augmentation: 75% success
- With augmentation: 82% success (7% improvement from virtual data)

### 2. Regularization

Prevent overfitting:

```python
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.Dropout(p=0.2),          # Drop 20% of activations randomly

    nn.Conv2d(64, 128, 3),
    nn.ReLU(),
    nn.Dropout(p=0.2),

    nn.Flatten(),
    nn.Linear(128*56*56, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),          # Stronger dropout before output

    nn.Linear(256, 8),
)

# Also use L2 regularization
loss = mse_loss + 0.0001 * sum(p.norm() for p in model.parameters())
```

### 3. Training Procedure

```python
# 1. Train/val split
train_data, val_data = split(demonstrations, ratio=0.8)

# 2. Learning rate schedule
initial_lr = 1e-3
optimizer = Adam(model.parameters(), lr=initial_lr)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# 3. Early stopping
best_val_loss = float('inf')
patience = 10
epochs_without_improvement = 0

for epoch in range(100):
    # Train one epoch
    train_loss = 0
    for batch in train_loader:
        images, actions = batch
        pred_actions = model(images)
        loss = MSE(pred_actions, actions)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validate
    val_loss = 0
    for batch in val_loader:
        images, actions = batch
        pred_actions = model(images)
        loss = MSE(pred_actions, actions)
        val_loss += loss.item()

    scheduler.step()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement > patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

---

## Deployment Considerations

### Latency Budget

Different tasks have different latency requirements:

```
Grasping:        100-500ms OK (slow motion allowed)
Pushing:         50-200ms OK
Fine manipulation: 20-50ms (fast response needed)
Reactive tasks:   < 10ms (emergency stop)

End-to-end network latency:
- CNN (small):    50ms  ✓ Good for most tasks
- CNN (large):    150ms ✓ Acceptable for slower tasks
- ViT (small):    200ms ✓ Slower but doable
- ViT (large):    500ms ✗ Too slow for reactive control
```

### Reliability & Safety

```python
# Prediction confidence
def infer_with_safety(image):
    logits = model(image)
    action = logits.mean
    uncertainty = logits.std

    # Only execute if confident
    if uncertainty < threshold:
        return action
    else:
        return safe_default_action  # E.g., open gripper, stop motion

# Monitor in production
def check_validity(action):
    # Check action is within bounds
    if not all(action.min() >= -1.0 and action.max() <= 1.0):
        return False, "Out of bounds"

    # Check action is not too different from previous
    if abs(action - previous_action).max() > 0.5:
        return False, "Too abrupt"

    return True, "OK"

# Fallback strategy
if not check_validity(action):
    action = previous_action  # Repeat last action
    if tries > 3:
        action = safe_default   # Give up, execute safe action
```

---

## Debugging End-to-End Policies

### Problem 1: Low Success Rate (30-50%)

```
Diagnosis:
1. Is this better than random (0%)?
   → If yes, network is learning
   → If no, fundamentally broken

2. Is training loss decreasing?
   → If no: learning rate too high, network broken
   → If yes: but test success low = distribution mismatch

3. Can network memorize training data?
   → Train to 95% on same episodes
   → If not: network too small
   → If yes: can generalize, data quality issue
```

### Problem 2: Catastrophic Failure (Falls off 90% to 20% after one failure)

```
This is the distribution mismatch problem:

Solution 1: DAgger-style approach
   Add failed trajectories to training data
   Retrain network
   Success improves

Solution 2: Use Diffusion Policy instead
   More robust to distribution shift

Solution 3: Ensemble prediction
   Train 3 separate networks on different data splits
   Predict action from all 3, use average
   If disagreement > threshold, use safe default
```

### Problem 3: Policy Works in Simulation, Fails on Real Robot

```
This is the sim-to-real gap:

Solution 1: Domain randomization in sim
   Randomize: physics, lighting, object appearance, camera noise
   Train on diverse environments
   Policy learns robust features

Solution 2: Fine-tune on real data
   Collect 50-100 real demonstrations
   Fine-tune network trained in sim
   Fast adaptation to real robot

Solution 3: Use RL for final tuning
   Start with BC in sim
   Fine-tune with RL on real robot (2-4 hours)
   Adapts to real dynamics
```

---

## Real-World Case Studies

### Case 1: Google RT-2 (55B Parameters)

**Architecture**:
```
Input: Image (top-down view)
       Natural language instruction (e.g., "Pick up the red cube")

Vision encoder: ViT-based
Language encoder: BERT-based
Fusion: Cross-attention + projection

Output: Action tokens (discrete action vocabulary)
```

**Training**:
- Pre-training: 130K robot demonstrations across 700+ tasks
- Fine-tuning: RL on selected tasks
- Multi-task learning: Same network for all tasks

**Results**:
- 97% success on 150 tasks it was trained on
- 55% success on zero-shot novel tasks
- Transfer learning: Fine-tune on 10 new tasks with 100 demos → 85% success

---

### Case 2: MIT Diffusion Policy

**Architecture**:
```
Input: Image (224×224)
       Action trajectory history (10 steps)

Encoder: ViT on current observation
Decoder: Diffusion model (50 steps)

Output: 10-step action sequence [a₀, a₁, ..., a₉]
```

**Training**:
- Data: 100-300 demonstrations per task
- Method: Diffusion-based trajectory generation
- Inference: 200-300ms (50 denoising steps)

**Results**:
- 80-90% success on 10 manipulation tasks
- vs BC: 50-60% success (33% improvement)
- Better generalization to novel objects

---

## Key Takeaways

✅ **End-to-end beats modular** - Single network learns robust mapping
✅ **Architecture matters** - CNN for simple tasks, ViT for complex reasoning
✅ **Multi-task helps** - Train on 10 tasks, transfer to new task faster
✅ **BC + RL is powerful** - Combine data efficiency and performance
✅ **Deployment is hard** - Latency, reliability, sim-to-real gaps
✅ **Data is the bottleneck** - Quality and quantity drive performance

---

## Navigation

- **[Back to Learning Overview →](intro.md)**
- **[Code Examples →](../static/code-examples/)**
- **[Exercises →](exercises.md)**

---

## Next Steps

1. **Choose your approach** - BC, Diffusion, or RL
2. **Collect demonstrations** - Quality > quantity
3. **Design architecture** - CNN for fast, ViT for complex
4. **Train and evaluate** - Compare methods on test set
5. **Deploy safely** - Add confidence thresholding and fallbacks

---

## Further Reading

- **End-to-End Learning for Self-Driving Cars** (Bojarski et al., 2016): Foundational paper
- **RT-1/RT-2** (Google, 2022-2023): Multi-task robotics at scale
- **Diffusion Policy** (MIT/UC Berkeley, 2023): Trajectory generation
- **Learning from Human Preferences** (OpenAI, 2017): RLHF for robotics

---

**Next Section:** [Practical Exercises →](exercises.md)

