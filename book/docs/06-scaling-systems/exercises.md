# Exercises: Production Robotics & Scaling Systems

Build and deploy production-quality robotic systems.

---

## Exercise 1: Implement Multi-Task Learning at Scale

**Scenario:** You're training a factory robot on 50 different assembly tasks simultaneously. Design a multi-task learning system that:
- Trains on all 50 tasks in one model
- Enables rapid fine-tuning of new tasks (10× speedup)
- Prevents catastrophic forgetting when adding new tasks

---

### Task 1A: Design the Architecture

**Question:** Sketch the architecture of a multi-task policy for 50 tasks.

What components would you need?
1. **Shared Encoder**: What should it encode?
   - Vision (RGB images from factory camera)
   - Proprioception (7 robot joint angles)

2. **Task Conditioning**: How do you tell the network which task to perform?
   - Embed task ID into fixed vector (32D)
   - Concatenate with encoder output

3. **Task-Specific Heads**: One per task or shared?
   - **Answer**: One output head per task (50 heads)
   - Each head: (256D encoder + 32D task embedding) → 4D action

**Architecture:**
```
Images (3×224×224) → Vision CNN (→256D)
                          ↓
Proprioception (7) → FC Encoder (→64D)
                          ↓
                    Fusion MLP (→256D)
                          ↓
                   Task Embedding (task_id → 32D)
                          ↓
                     Concatenate (256+32=288D)
                          ↓
              Task-Specific Head 0 → Action (4D)
              Task-Specific Head 1 → Action (4D)
              ...
              Task-Specific Head 49 → Action (4D)
```

---

### Task 1B: Implement Transfer Learning

**Question:** A new 51st task (a new assembly operation) needs to be added. How would you train it 10× faster than training from scratch?

**Solution:**

```python
# Start with pre-trained policy on 50 tasks
policy = MultiTaskPolicy(num_tasks=51)
policy.load_state_dict(torch.load('task_50_checkpoint.pt'))

# Collect only 50 demonstrations of new task (vs 500 normally)
new_task_data = collect_demonstrations(task_id=50, num_demos=50)

# Fine-tune ONLY the new task head, keep encoder frozen
policy.finetune_on_new_task(
    new_task_data=new_task_data,
    task_id=50,
    num_epochs=5,
    learning_rate=1e-4
)
```

**Why 10× faster?**
- Training from scratch: 500 demos × 5 epochs = 2,500 gradient updates
- Fine-tuning: 50 demos × 5 epochs = 250 gradient updates
- Encoder already learned generic "how to see assembly tasks"
- New head only learns "where to reach for this specific task"

**Catastrophic Forgetting Prevention:**

```python
# Mix old + new task data during fine-tuning
train_loader = DataLoader([
    *old_task_demonstrations,  # Keep learning task 0-49
    *new_task_data,             # Learn task 50
], batch_size=32)

for epoch in range(5):
    for batch in train_loader:
        pred_actions = policy(batch['images'], batch['proprioception'],
                            batch['task_ids'])
        loss = MSE(pred_actions, batch['actions'])
        loss.backward()
        optimizer.step()
```

This way, each update sees a mix of old/new tasks → gradual adjustment, no forgetting.

---

## Exercise 2: Design a 4-Robot Pilot Deployment

**Scenario:** You've proven the technology works in simulation. Now deploy on 4 real robots in a warehouse. Design the deployment strategy.

---

### Task 2A: Identify Real-World Challenges

**Question:** What could go wrong? List 5 challenges specific to real robot deployment.

**Challenge 1: Sim-to-Real Gap (Safety Critical)**
- Problem: 95% success in sim → 60% on real robot (gripper wear, lighting differences, object variation)
- Solution: Domain randomization during sim training
  ```python
  env = make_sim_env(
      randomize_gripper_friction=True,  # Real gripper wear
      randomize_lighting=True,           # Warehouse lighting variation
      randomize_object_texture=True,    # Worn/new objects
      randomize_table_height=True,      # Different workstations
  )
  ```

**Challenge 2: Latency Constraints**
- Problem: 200ms inference latency but need 100ms for real-time control
- Solution: Quantize model on robot, use smaller network
  ```python
  # Convert to 8-bit quantization for 3× speedup
  quantized_model = torch.quantization.quantize_dynamic(
      policy, {nn.Linear}, dtype=torch.qint8
  )
  # Deployment latency: 200ms → 70ms ✓
  ```

**Challenge 3: Robot Failures**
- Problem: Motor breaks, gripper jams → robot goes offline
- Solution: Detect failures, alert operator, continue on other robots
  ```python
  if robot.consecutive_failures > 5:
      alert_operator(f"Robot {robot.id} likely broken")
      robot.is_healthy = False
      continue_on_remaining_robots()
  ```

**Challenge 4: Network Connectivity**
- Problem: Warehouse WiFi drops → robots can't reach server
- Solution: Local fallback policy, buffer data locally
  ```python
  if not can_reach_server():
      use_local_policy()
      buffer_data_locally()
      # When network recovers, upload buffered data
  ```

**Challenge 5: Data Quality**
- Problem: Poor videos, labeling errors → garbage training data
- Solution: Automated data quality checks
  ```python
  def is_valid_trial(trial):
      # Check video has clear object
      assert video_blur_score < 0.2
      # Check action is reasonable
      assert action_magnitude < max_displacement
      # Check success label matches video
      assert success_label_consistent_with_video()
      return True
  ```

---

### Task 2B: Design the Hardware Setup

**Question:** Design the physical deployment for 4 robots. What infrastructure is needed?

**Answer:**

```
┌──────────────────────────────────────┐
│   Central Server (GPU Training)      │
│   ├─ RTX 3090 GPU                    │
│   ├─ 2TB SSD (data storage)          │
│   └─ Monitoring dashboard            │
└──────────────────────────────────────┘
           ↑    ↓
      WiFi / Ethernet
           ↑    ↓
  ┌────────┼────┼────────┐
  │        │    │        │
[Robot 1] [Robot 2] [Robot 3] [Robot 4]
  │        │    │        │
  └────────┼────┼────────┘
      Each has:
      - Edge GPU (RTX 2060)
      - 100GB local SSD
      - 1080p camera + gripper
      - Safety shutdown button
```

**Connectivity:**
- Wired: 100+ Mbps for video streaming
- WiFi: 50+ Mbps fallback
- 4G modem: Emergency connectivity
- Weekly sync: 7 hours (low network load)

---

## Exercise 3: Analyze Weekly Learning Progress

**Scenario:** You have 4-week pilot data. Analyze whether the system is improving.

---

### Task 3A: Interpret the Data

**Question:** You observe the following results:

```
Week 1: Fleet average 72% success
Week 2: Fleet average 74% success (+2%)
Week 3: Fleet average 75% success (+1%)
Week 4: Fleet average 76% success (+1%)
```

Is this good? What does it tell us?

**Answer:**

✓ **Good signs:**
- Consistent improvement every week
- Compounding: 72% → 76% = 5.5% absolute, 7.6% relative gain
- No regressions (would see success drop after bad model deployment)

⚠ **Concerning signs:**
- Diminishing returns (2% → 1% → 1% gain)
- 76% is still far from target (80%+)
- May be hitting saturation with current approach

**Why diminishing returns?**
1. **Data diversity limit**: Collected 700K trials/week, mostly seen variations
2. **Architecture limit**: Simple BC might not handle multi-modal grasps
3. **Domain gap**: Sim training still doesn't cover all real variations

**Next steps:**
- Switch to diffusion policy (handles multi-modality better) → expect 3-5% boost
- Add domain randomization specifically for observed failures → 2-3% boost
- Manual correction (expert labels for hardest 10% of cases) → 1-2% boost

---

### Task 3B: Failure Mode Analysis

**Question:** Examine failure logs and categorize. What's causing 24% of failures?

**Logs (Sample from 100 failed trials):**
```
25 failures: Gripper doesn't grip (too loose)
15 failures: Target object partially occluded (can't see it)
10 failures: Robot hits table edge (collision)
8 failures: Network latency (stale state)
7 failures: Object already removed by previous trial
5 failures: Other/unknown
```

**Answer:**

| Failure Type | Count | % | Root Cause | Solution | Priority |
|---|---|---|---|---|---|
| Gripper slip | 25 | 25% | Gripper wear after 10K cycles | Recalibrate every 1K trials | HIGH |
| Occlusion | 15 | 15% | Camera angle poor, object shadows | Add second camera | HIGH |
| Collision | 10 | 10% | Policy reaches into obstruction | Better safety constraints | MEDIUM |
| Latency | 8 | 8% | Wireless packet loss | Upgrade WiFi or go wired | LOW |
| Already removed | 7 | 7% | Trial sequencing issue | Better queueing | LOW |
| Unknown | 5 | 5% | Need video review | Manual inspection | MEDIUM |

**Cost-Benefit Analysis:**

```
Gripper recalibration:
  Cost: 2 hours/week
  Benefit: Fix 25% of failures = +6% success
  ROI: Massive

Second camera:
  Cost: $3K hardware + setup
  Benefit: Fix 15% of failures = +4% success
  ROI: Good (breaks even in 1 month)

Better safety constraints:
  Cost: 1 day engineer time
  Benefit: Fix 10% of failures = +2.4% success
  ROI: Excellent
```

**Recommended action:**
1. ✅ Do gripper recalibration this week (quick win)
2. ✅ Implement safety constraints (engineer task)
3. 💰 Plan second camera for next month (hardware budget)

---

## Exercise 4: Scale from 4 to 100 Robots

**Scenario:** The 4-robot pilot succeeded (80% success rate). Now scale to 100 robots across 10 facilities.

---

### Task 4A: Design the Federated Learning Pipeline

**Question:** With 100 robots, daily collection is 100K trials. How do you train efficiently?

**Answer:**

```
Daily Collection (100 robots × 1000 trials):
  Robot 1 → [1000 trials] ──┐
  Robot 2 → [1000 trials] ──┤
  ...                        ├→ Central Buffer (100K trials)
  Robot 100 → [1000 trials] ─┘

Weekly Sync (Sunday, all robots upload simultaneously):
  ├─ Data transfer: 100 robots × 10GB = 1TB total
  ├─ Time: ~6 hours on 10Gbps link
  └─ Storage: Data lake (S3/GCS)

Weekly Training (Sunday evening, 24 hours):
  ├─ Data: 700K new trials + 2.1M historical
  ├─ Infrastructure: 32-GPU cluster
  ├─ Method: Distributed data parallelism
  │  - Split 2.8M trials across 32 GPUs
  │  - Each GPU trains on ~88K trials
  │  - All-reduce averaging after each epoch
  ├─ Time: 24 hours for 50 epochs
  └─ Result: New model v24

Monday Deployment:
  ├─ A/B test: 50 robots get new model, 50 keep old
  ├─ Monitor: Compare success rate for 24 hours
  ├─ Decision:
  │  - If new > old: Deploy to all 100 robots
  │  - If new < old: Rollback, investigate why
  ├─ Time: 1 hour rollout
  └─ Impact: +0.2% success = +100 items/day!
```

**Communication Efficiency:**
```
Naive approach:
  - Robot 0 sends 10GB/week
  - Robot 1 sends 10GB/week
  - ...
  - Total bandwidth: 1TB/week = 19 Mbps sustained

Optimized approach (lossy compression):
  - H.264 video compression: 10GB → 2GB (-80%)
  - Downsample images: 1080p → 360p (-75% from compression)
  - Delta updates: Only send pixels that changed (-60%)
  - Result: 10GB → 0.5GB per robot
  - Total bandwidth: 50GB/week = 0.95 Mbps sustained ✓

Trade-off:
  - Loss of fine visual detail
  - But with modern networks, detail < representative data
  - 1000 trials with 360p better than 10 trials with 1080p
```

---

### Task 4B: Estimate Hardware Costs

**Question:** What's the total infrastructure cost for 100-robot fleet?

**Answer:**

```
CAPITAL COSTS:

Robot Hardware:
  - 100 × Robot arm: $50K/unit = $5M
  - 100 × Gripper: $5K/unit = $500K
  - 100 × Camera + sensors: $3K/unit = $300K
  - 100 × Edge GPU (RTX 2060): $2K/unit = $200K
  Subtotal: $6M

Central Infrastructure:
  - 32 × V100 GPU: $8K/unit = $256K
  - High-end servers (CPU): $100K
  - Storage (50TB SSD): $200K
  - Networking (10Gbps): $50K
  Subtotal: $606K

Facilities:
  - 10 facilities with power/WiFi/tables: $500K
  Subtotal: $500K

TOTAL CAPEX: $7.1M

ANNUAL OPERATING COSTS:

Personnel:
  - ML Engineers (4): $600K
  - Roboticists (10): $800K
  - DevOps/Operations (4): $500K
  - Interns/labeling: $100K
  Subtotal: $2M

Infrastructure:
  - Electricity (100 robots × 5KW × 8h/day): $150K
  - Maintenance/parts: $300K
  - Cloud storage (50TB to cloud): $50K
  - Network: $50K
  Subtotal: $550K

TOTAL OPEX: $2.55M/year

FINANCIAL ANALYSIS:

Revenue per item: $50 (assuming factory produces high-value items)
Throughput: 5000 items/day × 250 working days = 1.25M items/year
Annual revenue: 1.25M × $50 = $62.5M

Profit (Year 1): $62.5M - $2.55M = $60M
Profit (Year 2+): $60M (capex amortized over 5 years)

ROI: (60M / 7.1M) × 100 = 845% Year 1 ✓✓✓
Payback period: 6 weeks (!!)
```

---

## Challenge Exercises

**Challenge 1: Federated Learning with Heterogeneous Data**

Different facilities have different tasks (warehouse A does grasping, warehouse B does insertion). How do you train one global model that works everywhere?

**Approach:**
- Task-weighted loss: `loss = sum(w_task × loss_task for all tasks)`
- Assign weights inversely to task difficulty
- Harder tasks (insertion) get higher weight during training
- Result: Model improves on hard tasks without hurting easy ones

---

**Challenge 2: Online Model Rollback**

A new model v25 is deployed but causes 10% success rate drop (80% → 70%). You have 100 robots and 1 hour to detect and fix.

**Steps:**
1. **Detect** (5 min): Automated anomaly detection sees success rate below threshold
2. **Alert** (1 min): Send alert to on-call engineer
3. **Evaluate** (20 min): Engineer checks failure logs, confirms bad model
4. **Rollback** (2 min): Send rollback command to all robots
5. **Verify** (10 min): Confirm success rate returns to 80%
6. **Investigate** (30 min): Why did model v25 fail? Training bug? Data issue?

**Result:** 1 hour max downtime, minimal impact to operations

---

**Challenge 3: Simulation Fidelity Trade-offs**

You need to choose between:
- **Option A**: Gazebo (free, realistic physics, slow)
  - Collect 100 demos: 24 hours real time
  - Cost: Robot time

- **Option B**: Isaac Sim (expensive, ultra-realistic, fast)
  - Collect 100 demos: 30 minutes wallclock time
  - Cost: $500/month GPU

Which is cost-effective? How many demos do you need before Isaac wins?

**Answer:**

```
Cost per demo:

Option A (Gazebo):
  Robot time: 24 hours / 100 demos = 14.4 min/demo
  Robot cost: $70/hour = $16.80/demo

Option B (Isaac Sim):
  Wallclock: 30 min / 100 demos = 18 sec/demo
  GPU cost: $500/month ÷ 720 hours ÷ 60 = $0.01/min
  Cost: 18 sec × $0.01/min ÷ 60 = $0.003/demo

Break-even:
  When do total costs match?
  Gazebo: $16.80/demo × N
  Isaac: $0.003/demo × N + $500 GPU

  $16.80N = $0.003N + $500
  $16.797N = $500
  N = 30 demos

  After 30 demos, Isaac wins (amortize GPU cost)
  For 100 demos: Gazebo $1680 vs Isaac $500 ✓
```

---

## Further Reading

**Real-World Deployment:**
- Boston Dynamics: Spot fleet coordination (unpublished, internal)
- Tesla: Humanoid manufacturing (Musk, 2023)
- Amazon: Robotic fulfillment economics (Banker, 2022)

**Federated Learning:**
- "Federated Learning: Challenges, Methods, and Future Directions" (Kairouz et al., 2021)
- "Communication-Efficient Learning of Deep Networks from Decentralized Data" (McMahan et al., 2016)

**Scaling Systems:**
- "Scaling Language Models: A Guide to Transformer Efficiency" (Hoffmann et al., 2022)
- "NVIDIA Isaac Sim for Robot Simulation" (official docs)

**Cost Analysis:**
- "The Economics of Automation" (Acemoglu & Johnson, 2018)
- "DHL Logistics Trend Radar" (includes automation ROI)

---

## Next Steps

After completing these exercises, you should be able to:
- ✅ Design multi-task systems for 100+ tasks
- ✅ Analyze sim-to-real gaps and design mitigations
- ✅ Build federated learning systems for 100+ robots
- ✅ Estimate costs and ROI for production deployments
- ✅ Diagnose failures and improve iteratively

**Ready for production?** Move on to Phase 9: Real-World Deployment & Capstone.

