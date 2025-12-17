# Multi-Task Learning: Training at Scale

Training on 150+ manipulation tasks simultaneously.

---

## The Multi-Task Problem

### Single-Task Learning

```
Dataset 1 (grasping):  500 demos  → Network 1  → 85% success
Dataset 2 (pushing):   400 demos  → Network 2  → 80% success
Dataset 3 (insertion): 300 demos  → Network 3  → 75% success

Problem:
  ✗ Three separate models (3× storage, 3× inference time)
  ✗ No knowledge sharing
  ✗ Adds new task = retrains from scratch
```

### Multi-Task Learning

```
Dataset 1 + 2 + 3 (1200 demos total)
                      → Single Network → 80% on grasping
                                      → 76% on pushing
                                      → 72% on insertion

Benefit:
  ✓ One model (1× storage, 1× inference time)
  ✓ Knowledge sharing (visual features help all tasks)
  ✓ Transfer learning (new task faster)
  ✓ Scaling (easier to add task 150)
```

---

## Architecture: Multi-Task Networks

### Basic Approach: Task Conditioning

```
Input: observation (image, proprioception)
       task_id (which task: 0=grasp, 1=push, 2=insert, etc.)

Shared Encoder:
  Vision: image → CNN → 256D features
  Proprioception: joint angles → FC → 64D
  Concatenate: 256+64 = 320D

Task Embedding:
  task_id → lookup table → 32D embedding

Fusion:
  [shared_features(320D), task_embedding(32D)] → 352D

Shared Head:
  FC: 352D → 256D → 128D

Task-Specific Heads:
  Grasp head:    128D → 4D (x, y, z, width)
  Push head:     128D → 2D (direction, force)
  Insert head:   128D → 3D (depth, angle, speed)
  ...
  Task 150 head: 128D → (action_dim for that task)
```

### Loss Function: Task-Weighted

```python
loss_total = Σ w_i * loss_i

where:
  w_i = weight for task i
  loss_i = MSE(predicted_action_i, expert_action_i)

Weighting strategies:

1. Equal weight:
   w_i = 1 for all tasks
   Problem: High-loss tasks dominate

2. Inverse loss:
   w_i = 1 / loss_i (recent batch)
   Effect: Focus on hard tasks

3. Sample-based:
   w_i = num_demos_i / total_demos
   Effect: More data = more weight
```

### Real Example: RT-2 Architecture

**Google's robotics transformer** (Brohan et al., 2023):

```
Input: Vision + Language + Proprioception

Vision Encoder:
  ViT-4B (4 billion parameters)
  Trained on internet images → understands any object

Language Encoder:
  BERT + fine-tuned on robot instructions
  "Pick up the red cube" → 1024D embedding

Proprioception:
  Joint positions (7D) → FC → 64D
  Previous actions (4D) → FC → 32D

Fusion (Cross-Attention):
  Vision queries attend to language
  Understand which language words matter for this scene

Action Decoder:
  1024D (fusion output) → action tokens
  Trained on 150+ tasks

Results:
  - 97% on training tasks (seen during training)
  - 55% on novel tasks (zero-shot transfer)
  - 85% after fine-tuning on 10 examples
```

---

## Training Multi-Task Networks

### Dataset Organization

```
```
Total: 100K demonstrations across 150 tasks

Task distribution:
  Grasping (20 variants):     30K demos
  Pushing (15 variants):      20K demos
  Insertion (10 variants):    15K demos
  Manipulation (105 others):  35K demos

Loading strategy:
  Batch size: 128 samples
  ├─ 30% from grasping (30K / 100K)
  ├─ 20% from pushing
  ├─ 15% from insertion
  └─ 35% from others

Result: Model sees balanced task distribution
```

### Training Loop

```python
for epoch in range(100):
    epoch_loss = 0

    for batch in data_loader:
        # batch contains samples from all 150 tasks
        images = batch['images']
        task_ids = batch['task_ids']
        actions = batch['actions']

        # Shared forward pass
        shared_features = encoder(images)  # (B, 320)

        # Task-specific predictions
        task_embeddings = task_embedding_table[task_ids]  # (B, 32)
        combined = cat([shared_features, task_embeddings])  # (B, 352)

        # One network, shared + task-specific heads
        action_pred = network(combined, task_ids)  # (B, action_dim)

        # Loss per task (weights based on task difficulty)
        loss = compute_weighted_loss(action_pred, actions, task_ids)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {epoch_loss / len(data_loader):.4f}")
```

### Hyperparameter Tuning

```
Challenge: What learning rate works for 150 tasks?

Single-task:
  LR = 1e-3 works fine (converges in 50 epochs)

Multi-task:
  LR = 1e-3 is too high (oscillates)
  LR = 1e-4 is too low (converges slowly)
  LR = 1e-3.5 (3.16e-4) works best

Why? 150 tasks = 150× gradient noise
     Need smaller learning rate for stability

Solution: Learning rate schedule
  Initial: 1e-3.5
  Warm-up: Increase to 1e-3 over first 10 epochs
  Decay: Decrease to 1e-4.5 over last 50 epochs
```

---

## Transfer Learning: Pre-Train & Fine-Tune

### Why Pre-Training Helps

```
Scenario 1: Train from scratch on new task
  Start with random weights
  Random → need 1000 trials to 70% success
  Timeline: 3-4 weeks
  Success: 75%

Scenario 2: Transfer from RT-2 pre-training
  Start with weights from 150 tasks
  Already understand: grasping, object detection, pushing
  Need 100 trials to 70% success
  Timeline: 3-4 days
  Success: 85% (better!)

Speed improvement: 10× faster
Performance improvement: +10% success
```

### Fine-Tuning Strategy

```
Step 1: Load pre-trained weights
  network = load_pretrained('rt2_weights.pth')

Step 2: Freeze shared encoder
  for param in network.shared_features.parameters():
      param.requires_grad = False

  Why? Vision features (edges, objects) transfer across tasks
  Freezing prevents overfitting to new task's small dataset

Step 3: Train task-specific head
  for param in network.task_head[new_task].parameters():
      param.requires_grad = True

  Only 50K parameters trainable (vs 1B in full network)
  Much faster training, less overfitting

Step 4: Fine-tune later layers (optional)
  if not converging after 1000 steps:
      unfreeze fusion layer
      reduce learning rate 10×
      train for 500 more steps

Result:
  Epochs 1-10: Head training (quick)
  Epochs 11-20: Fusion layer fine-tuning (if needed)
  Stop if test success plateaus
```

### Real Results: Transfer Learning Impact

**Study**: Training new task with/without pre-training

| Setup | Data Needed | Training Time | Success Rate |
|-------|------------|---------------|-------------|
| From scratch | 500 demos | 2-3 weeks | 75% |
| Pre-trained scratch | 500 demos | 2-3 weeks | 82% |
| Pre-trained, fine-tune | 100 demos | 3-4 days | 84% |
| Pre-trained, fine-tune | 50 demos | 2 days | 78% |

**Key insight**: Pre-training + fine-tuning gets to 80%+ in days (vs weeks from scratch)

---

## Continual Learning: Adding New Tasks

### The Catastrophic Forgetting Problem

```
Step 1: Train on Tasks 1-100 (grasp, push, insertion, etc.)
  Success on all 100: 80% average

Step 2: Fine-tune Task 101 (new grasping variant)
  Add 200 new examples, train for 5 epochs
  Task 101 success: 85% (good!)
  But... Task 1-100 success: 45%! (catastrophic!)

Problem: Network weights adapted to Task 101,
         forgot knowledge from Tasks 1-100
```

### Solution 1: Rehearsal Learning

```python
def continual_learning(new_task_data, old_task_data):
    # Mix old and new data
    for epoch in range(10):
        # 50% batch from new task
        # 50% batch from old tasks (random sample)

        batch = combine([
            sample_new_task(100),     # 50% new
            sample_old_tasks(100),    # 50% old (from all 100)
        ])

        # Train on mix
        loss = network(batch)
        loss.backward()
        optimizer.step()

Result:
  Task 101 success: 82%
  Tasks 1-100 success: 79% (preserved!)
```

### Solution 2: Task-Specific Parameters

```
Share features, but each task has its own small head:

Shared Encoder: (shared across ALL tasks)
  Vision CNN → 256D features
  Proprioception FC → 64D
  Total: 2M parameters

Task-Specific Head (per task):
  Task 1 head: 100 parameters
  Task 2 head: 120 parameters
  ...
  Task 150 head: 90 parameters
  Task 151 head: 110 parameters (NEW)

Adding Task 151:
  ✓ Shared encoder unchanged (no forgetting)
  ✓ Add small new head (quick training)
  ✓ All old tasks work as before
```

---

## Architectural Patterns

### Pattern 1: Separate Heads (Simplest)

```
One shared encoder + one head per task
  Pros: Simple, no forgetting
  Cons: Doesn't scale (150 heads = overhead)
  Best for: 10-30 tasks
```

### Pattern 2: Task Conditioning (Popular)

```
One shared encoder + task embedding + shared head
  Pros: Scales well (task embedding learned)
  Cons: Slight performance drop per task
  Best for: 30-500 tasks
```

### Pattern 3: Mixture of Experts

```
One shared encoder + multiple expert networks
  Router network: input → which expert?
  Expert 1: Handles grasping + reaching
  Expert 2: Handles pushing + insertion
  ...

  Pros: Flexible, can specialize
  Cons: Complex, hard to train
  Best for: 100+ diverse tasks
```

---

## Scaling Laws: Performance with More Tasks

```
Success rate vs number of tasks:

100% ├─ (single task)
     │
  95% ├────●
     │     \
  90% ├──────●─────
     │       \     \
  85% ├────────●────●──
     │         \    \  \
  80% ├─────────●────●──●──
     │          \    \  \  \
  75% ├──────────●───●───●──●──
     │
  70% └────────────────────────→
        1  10  50  100 150 300
        Number of tasks

Pattern:
  1 task:    90% (single-task network)
  10 tasks:  87% (shared encoder helps)
  50 tasks:  82% (diminishing returns)
  100 tasks: 80% (typical plateau)
  150 tasks: 79% (RT-2 actual result)

Explanation:
  - Task diversity reduces performance per task
  - But... ONE model instead of 150 models
  - Trade-off is worth it for production systems
```

---

## Multi-Task Evaluation

### Per-Task Metrics

```
For each of 150 tasks:
  Success rate: % of test trials succeeded
  Confidence: Network's output uncertainty
  Transfer ratio: Performance on novel objects

Report:
  Grasping: 82% (20 variants: 75-88%)
  Pushing:  78% (15 variants: 70-85%)
  Insertion: 76% (10 variants: 68-82%)
  Other:     77% (105 tasks: 65-89%)

  Average: 79%
  Min: 65% (hardest task)
  Max: 89% (easiest task)
```

### Failure Analysis

```
Which tasks fail most often?

Grasping soft objects:     15% failure (too much variation)
Long-horizon tasks:        18% failure (many steps = errors accumulate)
Rare object types:         22% failure (few training examples)
High-precision tasks:      12% failure (fine motor control)

Mitigation:
  - Collect more data on hard tasks
  - Use RL to improve long-horizon
  - Domain randomization for rare objects
```

---

## Key Takeaways

✅ **Task Conditioning**: Embed task ID, share encoder, task-specific heads
✅ **Loss Weighting**: Balance tasks by difficulty or data quantity
✅ **Transfer Learning**: Pre-train 150 tasks, fine-tune new task in days
✅ **Continual Learning**: Mix old/new data to avoid forgetting
✅ **Scaling Laws**: Performance drops 1-2% per 10× more tasks
✅ **Flexibility**: Add new tasks without retraining entire network

---

## Real-World Deployment

**Google RT-2 Deployment**:
- One 55B model for 150+ tasks
- Uses condition on language instruction
- Inference: 0.5-1 second per prediction
- Deployed on real robots worldwide

**Meta/Fair Embodied AI**:
- Multi-task pre-training on 100 tasks
- Transfer to new environments, new objects
- 85%+ success after fine-tuning

---

## Next Steps

1. **Understand task conditioning** - Embed task ID in network
2. **Implement loss weighting** - Balance tasks during training
3. **Try transfer learning** - Pre-train → fine-tune for new task
4. **Add continual learning** - Mix old/new data in batches
5. **Benchmark** - Measure per-task performance

---

## Further Reading

- **Multi-Task Learning** (Ruder, 2017): Comprehensive overview
- **RT-1** (Brohan et al., 2022): First large-scale multi-task robot learning
- **RT-2** (Brohan et al., 2023): Language-conditioned multi-task at scale
- **Continual Learning in Robots** (Rusu et al., 2016): Avoiding catastrophic forgetting

---

**Next Section:** [Real Robot Deployment →](real-robot-deployment.md)

