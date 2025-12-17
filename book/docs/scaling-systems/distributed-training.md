# Distributed Training: Multi-GPU & Multi-Robot Systems

Scaling training from single GPU to clusters and robot fleets.

---

## The Scaling Challenge

### Single GPU Training

```
Setup:
  1 GPU (RTX 3090)
  Batch size: 64
  Model: 150M parameters

Timeline for training on 150 tasks:
  Data: 100K demonstrations
  Throughput: 10K samples/hour
  Total time: 10 hours of GPU training
  Epochs: 20 epochs → 200 hours
  Real time: ~2-3 weeks (including I/O, validation)
```

### Multi-GPU Training (Naive)

```
Setup:
  8 GPUs (RTX 3090s)
  Batch size: 64 × 8 = 512 (per GPU: 64)

Expected: 8× speedup = 25 hours (3 days)

Reality:
  Communication overhead: 15%
  I/O bottleneck: 20% (disk can't keep up)
  Batch normalization issues: 5% accuracy drop
  Actual speedup: 6× → 33 hours (still 2 days)

Better approach: Use distributed data parallelism
```

### Distributed Training Benefits

```
Single GPU:     2-3 weeks to train 150 tasks
Multi-GPU (8):  3-4 days
Multi-node (32 GPUs): 1-2 days
Fleet (100 robots): Hours to days
```

---

## Data Parallelism: The Standard Approach

### How It Works

```
         Data Split 1 (25K samples)
                ↓
       ┌────────────────────┐
       │                    │
    GPU 0                 GPU 1
    Batch 1              Batch 2
    Forward pass         Forward pass
    Backward pass        Backward pass
    Gradients 1          Gradients 2
       │                    │
       └────────────────────┘
            All-reduce
            Average gradients
            Update shared weights
                ↓
            Both GPUs now have
            same updated weights
```

### Implementation with PyTorch

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed training
dist.init_process_group(backend='nccl')  # NVIDIA Collective Communications Library
rank = dist.get_rank()
world_size = dist.get_world_size()

# Create model and move to GPU
device = torch.device('cuda', rank)
model = load_model().to(device)

# Wrap model for distributed training
model = DDP(model, device_ids=[rank])

# Create sampler that splits data across GPUs
from torch.utils.data import DistributedSampler
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    sampler=sampler,  # Ensures no overlap between GPUs
    num_workers=4,
)

# Training loop (same as single GPU!)
for epoch in range(100):
    sampler.set_epoch(epoch)  # Reshuffle data

    for batch in dataloader:
        images, actions = batch
        images = images.to(device)
        actions = actions.to(device)

        # Forward pass
        pred = model(images)
        loss = F.mse_loss(pred, actions)

        # Backward pass (gradients averaged across GPUs automatically)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log only from rank 0 (avoid duplicate logging)
        if rank == 0:
            print(f"Loss: {loss.item():.4f}")

dist.destroy_process_group()
```

---

## Communication Efficiency: The Bottleneck

### Gradient Synchronization Cost

```
Each GPU update:
  1. Forward pass: 100ms
  2. Backward pass: 150ms
  3. All-reduce gradients: 50ms ← COMMUNICATION OVERHEAD
  Total: 300ms

With 8 GPUs:
  Forward: 100ms (shared across 8)
  Backward: 150ms (shared across 8)
  Communication: 50ms (SAME AS BEFORE!)

Communication is not reduced by more GPUs!
```

### Strategies to Reduce Communication

**Strategy 1: Gradient Accumulation**
```python
accumulation_steps = 4

for batch_idx, batch in enumerate(dataloader):
    # Forward/backward but don't sync yet
    loss = model(batch)
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        # Sync gradients only every 4 steps
        optimizer.step()
        optimizer.zero_grad()

Result:
  Communication: Every 4 steps instead of every step
  4× reduction in communication overhead
  Effective batch size: 64 × 4 = 256
```

**Strategy 2: Gradient Compression**
```python
# Before all-reduce: 1 billion float32s = 4GB
# Communication time: 50ms (on 100Gbps network)

# Compress to lower precision
compressed_grads = quantize_to_int8(gradients)  # 1/4 size = 1GB
# Communication time: 12.5ms (4× faster)

# Decompress after all-reduce
full_grads = dequantize_to_float32(compressed_grads)

Result:
  Communication: 4× faster
  Accuracy: Negligible difference
```

**Strategy 3: Federated Learning (Multi-Robot)**
```
Instead of continuous gradient sync:

Each robot trains locally:
  Robot 1: Train 10K samples locally → local model
  Robot 2: Train 10K samples locally → local model
  ...
  Robot 100: Train 10K samples locally → local model

Every N hours, average models:
  Global model = Average(Robot 1-100 models)
  Send back to all robots for next phase

Result:
  Communication: Only once per training phase (not per batch)
  100× reduction in communication
  Trade-off: Slower convergence (eventual consistency)
```

---

## Multi-Robot Fleet Training

### Architecture

```
┌─────────────────────────────────────────────────┐
│              Central Learning Server             │
│  ├─ Aggregates data from all robots             │
│  ├─ Trains global model                         │
│  └─ Broadcasts updates to robots                │
└─────────────────────────────────────────────────┘
        ↑                                ↓
        │ Upload collected data         │ Download new weights
        │ (1GB/day per robot)           │ (50MB/week)
        │                               │
  ┌─────┴─────────────────────────────┴──────┐
  │                                            │
Robot 1 (Warehouse A)    Robot 2 (Office B)    ...    Robot 100
├─ Local policy          ├─ Local policy               ├─ Local policy
├─ Collect grasps       ├─ Collect pushing            ├─ Collect insertion
└─ Store 1000 trials    └─ Store 800 trials          └─ Store 1200 trials
```

### Federated Learning Pipeline

```python
class RobotFleet:
    def __init__(self, num_robots=100, sync_interval=24):
        self.robots = [Robot(i) for i in range(num_robots)]
        self.global_model = load_base_model()
        self.sync_interval = sync_interval  # hours

    def run_async_training(self):
        """Robots train independently, periodically sync."""

        while True:
            # Phase 1: Local training (24 hours)
            for robot in self.robots:
                # Each robot:
                # 1. Collect 1000 trials
                collected_data = robot.collect_trials(num_trials=1000)

                # 2. Train locally
                robot.local_model = robot.finetune(
                    self.global_model,
                    collected_data,
                    epochs=10,
                )

            # Phase 2: Synchronization
            # 1. Collect all models from robots
            all_models = [robot.local_model for robot in self.robots]

            # 2. Average weights
            self.global_model = average_models(all_models)

            # 3. Send back to all robots
            for robot in self.robots:
                robot.update_model(self.global_model)

            # 4. Log progress
            self.log_performance()

    def log_performance(self):
        """Measure improvement across fleet."""
        for robot in self.robots:
            success_rate = robot.evaluate(num_trials=50)
            print(f"Robot {robot.id}: {success_rate*100:.1f}%")

        fleet_average = np.mean([r.evaluate(50) for r in self.robots])
        print(f"Fleet average: {fleet_average*100:.1f}%")
```

### Benefits of Fleet Training

```
Before Fleet Learning:
  Robot 1: Trains alone on grasping
    Week 1: 40% success (random policy)
    Week 2: 65% success
    Week 3: 78% success
    Week 4: 82% success (plateau)

After Fleet Learning:
  Robot 1: Trains with 99 others
    Week 1: 40% (local training) + 50% (fleet knowledge)
    Week 2: 65% (local) + 68% (fleet → shared learning!)
    Week 3: 78% (local) + 85% (fleet → leverage all robots!)
    Week 4: 82% (local) + 90% (fleet → collective intelligence!)

Improvement:
  Without fleet: 82% (single robot learning)
  With fleet: 90% (+10%, from 99 other robots!)
```

---

## Communication Efficiency Across Robots

### The Challenge

```
Naive approach: Send full 500MB model every sync

Robot 1 → Central Server: 500MB × 100 robots = 50GB upload
Central Server → All robots: 50GB download

Network: 1Mbps (typical warehouse WiFi)
Time: 50GB = 50 billion bits / 1Mbps = 50,000 seconds = 14 hours!

Problem: Sync takes longer than local training!
```

### Solution 1: Delta Updates (Only Send Changes)

```python
def compute_delta(old_model, new_model):
    """Send only weight changes, not full model."""
    delta = {}
    for name, param in new_model.named_parameters():
        old_param = old_model.state_dict()[name]
        change = param - old_param

        # Only send if change is significant
        if change.norm() > threshold:
            delta[name] = change

    return delta

# Instead of 500MB, maybe 50MB (10× compression!)
# Time: 50 seconds (acceptable)
```

### Solution 2: Model Distillation (Smaller Model)

```python
# Full model: 500M parameters, 500MB
# Distilled model: 50M parameters, 50MB
# Accuracy: 95% of full model

# Training distilled model:
# On server, train large model (slow)
# Distill to small model (fast)
# Send small model to robots (10 seconds)

# On robots:
# Inference is 10× faster (100ms instead of 1s)
# Bandwidth is 10× less
```

### Solution 3: Differential Parameter Updates

```
Instead of averaging models:

Local updates (each robot):
  w1_local = SGD(w1_global, local_data)
  delta1 = w1_local - w1_global

Aggregate only deltas (not models):
  global_delta = average(delta1, delta2, ..., delta100)
  w_new = w_global + learning_rate × global_delta

Benefits:
  - Communication: Only deltas (10× smaller)
  - Convergence: Similar to averaging
  - Flexibility: Works with different local architectures
```

---

## Distributed Training Challenges

### Challenge 1: Stragglers (Slow Robots)

```
Scenario: 100 robots, one has slow internet

Expected: All robots sync every 24 hours
Reality: 99 robots ready, 1 still uploading (takes 36 hours)

Problem: Have to wait for slowest robot (synchronous aggregation)

Solutions:
  1. Async update: Don't wait, use available models (eventually consistent)
  2. Early stopping: Kick out robots that fall behind
  3. Compression: Reduce data size so everyone finishes
```

### Challenge 2: Non-IID Data (Different Distributions)

```
Scenario: Fleet of 100 robots

Robot 1 (Office): Sees small objects, clean tables
Robot 50 (Warehouse): Sees large boxes, dirty surfaces
Robot 100 (Factory): Sees precision parts, clean workspace

Their data is NOT identically distributed!
But federated learning assumes IID data.

Problem:
  Global model trained on mix of all data
  Optimal for average, suboptimal for each robot

Solution:
  Personalized federated learning
  Each robot keeps local head (while sharing encoder)
  Result: Better fit for each environment
```

### Challenge 3: Network Failures

```
Scenario: Network goes down for 2 hours

What happens:
  ✓ Local training continues (robots are autonomous)
  ✗ No global synchronization
  ✓ When network recovers, sync catches up

Result:
  Robots continue improving locally
  Updates delay by ~2 hours
  No catastrophic failures
  (This is actually a feature of federated learning!)
```

---

## Scaling Laws: Speed vs Fleet Size

```
Training time vs number of GPUs:

Time (hours)
    100 ├─ (1 GPU)
        │
     50 ├──● (2 GPUs, 1.9× speedup)
        │   \
     25 ├────●──── (4 GPUs, 3.8× speedup)
        │     \
     12 ├──────●─── (8 GPUs, 8.0× speedup - close to ideal!)
        │       \
      6 ├────────●──● (16 GPUs, 15× speedup - communication limits)
        │         \ \
      3 ├─────────●──●──● (32 GPUs, 25× - diminishing returns)
        │
      1 └──────────────────→
        1   2   4   8  16  32
        Number of GPUs

Pattern:
  1-8 GPUs: Near-linear scaling (8× speedup with 8 GPUs)
  8-32 GPUs: Communication overhead becomes visible (25× with 32)
  32+ GPUs: Scales poorly unless communication-optimized

Lesson: 8-16 GPUs is sweet spot for most problems
```

---

## Monitoring Distributed Training

### Metrics to Track

```python
class DistributedMetrics:
    def __init__(self):
        self.metrics = {
            'loss': [],
            'throughput_samples_per_sec': [],
            'communication_time_ms': [],
            'compute_time_ms': [],
            'synchronization_efficiency': [],  # 1.0 = perfect
        }

    def compute_efficiency(self):
        """How much time spent on communication vs computation?"""
        if compute_time + comm_time == 0:
            return 0
        return compute_time / (compute_time + comm_time)

    def print_summary(self):
        """Print training efficiency."""
        eff = self.compute_efficiency()
        print(f"Synchronization efficiency: {eff*100:.1f}%")

        if eff < 0.7:
            print("⚠️  WARNING: Spending too much time communicating!")
            print("   Suggestion: Use gradient accumulation or compression")
```

---

## Production Deployment: Distributed Training Best Practices

### Setup Checklist

- [ ] Test on single GPU first (verify correctness)
- [ ] Test on 2 GPUs locally (verify synchronization)
- [ ] Test on multi-node cluster (verify networking)
- [ ] Measure speedup (should be >6× on 8 GPUs)
- [ ] Monitor for stragglers (slow robots/GPUs)
- [ ] Set up checkpointing (save model every epoch)
- [ ] Implement fault tolerance (resume after failure)

### Resource Requirements

```
For training on 100K samples across 150 tasks:

Single GPU:
  GPU memory: 24GB (RTX 3090)
  Training time: 200 hours
  Cost: $50/month (used GPU)

8-GPU cluster:
  GPU memory: 24GB × 8 = 192GB
  Training time: 25 hours
  Cost: $400/month + networking
  Speedup: 8×, Cost increase: 8×

16-GPU cluster:
  GPU memory: 384GB
  Training time: 15 hours
  Cost: $800/month
  Speedup: 13×

Multi-robot fleet (100 robots):
  Total compute: Distributed (1 GPU per robot)
  Training time: 24 hours (async)
  Cost: $10K/month (robot hardware)
  Speedup: Effective 50× (all robots in parallel)
  Advantage: Cheap bandwidth, no single point of failure
```

---

## Real-World Example: Google RT-2 Training

```
Scale:
  - 150+ tasks
  - 100K+ demonstrations
  - 55B parameters

Distributed setup:
  - TPU cluster (thousands of TPUs)
  - Data parallelism across TPUs
  - Synchronous gradient averaging

Training:
  - Time: Weeks of TPU cluster time
  - Cost: $millions in compute
  - Result: 97% on training tasks, 55% zero-shot

Lessons:
  - Distributed training is essential at this scale
  - Still took weeks (even with massive compute)
  - Zero-shot transfer is hard (55% vs 97%)
  - Fine-tuning helps (85% with 10 examples)
```

---

## Key Takeaways

✅ **Data Parallelism**: Scale from 1 GPU to 100+ GPUs
✅ **Communication Efficiency**: Use compression, accumulation, deltas
✅ **Federated Learning**: Train on distributed robots without central server
✅ **Fault Tolerance**: System continues if robots/GPUs fail
✅ **Monitoring**: Track compute/communication time ratio
✅ **Sweet Spot**: 8-16 GPUs before communication dominates

---

## Next Steps

1. **Understand all-reduce**: How gradients are averaged across GPUs
2. **Implement data parallelism**: PyTorch DDP on 2-4 GPUs
3. **Measure efficiency**: Track communication vs computation
4. **Implement gradient accumulation**: Reduce communication by 4×
5. **Deploy fleet training**: Async updates across 10+ robots

---

## Further Reading

- **Distributed Machine Learning** (Verbraeken et al., 2021)
- **Communication-Efficient Learning** (McMahan et al., 2017): Federated Learning
- **Efficient Large-Scale Language Model Training** (Chowdhery et al., 2022): PaLM
- **Asynchronous Distributed Training** (Wang et al., 2020)

---

**Next Section:** [Benchmarking at Scale →](benchmarking-framework.md)

