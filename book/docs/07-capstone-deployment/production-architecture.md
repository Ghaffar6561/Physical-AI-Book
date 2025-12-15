# Section 1: Production System Architecture

Complete system design for deploying 100+ robots in real factories.

---

## Overview: From Research to Production

### The Gap

```
RESEARCH PROTOTYPE:
  1 robot
  1 task
  80% success
  Unlimited time to debug
  Single person operation

PRODUCTION SYSTEM:
  100+ robots
  150+ tasks
  98%+ success required
  Minutes to detect and fix
  24/7 autonomous operation
  Data-driven improvement loop
```

### The Missing Pieces

A research system is missing:
1. **Monitoring** - How do you know what 100 robots are doing?
2. **Orchestration** - How do you schedule tasks across robots?
3. **Integration** - How does this connect to factory systems?
4. **Recovery** - What happens when things fail?
5. **Safety** - How do you ensure operators and equipment are safe?
6. **Compliance** - How do you meet regulatory requirements?

---

## Production Architecture (5-Layer Model)

```
┌────────────────────────────────────────────────────┐
│ Layer 5: HUMAN INTERFACE                           │
│ ├─ Operator Dashboard (web/mobile)                 │
│ ├─ Task Submission (API, manual, automated)        │
│ └─ Incident Response (alerts, runbooks)            │
└────────────────────────────────────────────────────┘
  ↑                                                  ↓
┌────────────────────────────────────────────────────┐
│ Layer 4: ORCHESTRATION & CONTROL                   │
│ ├─ Task Scheduler (queue, prioritization)          │
│ ├─ Fleet Manager (robot allocation)                │
│ ├─ Model Rollout Controller (A/B testing)          │
│ └─ Safety Monitor (constraint checking)            │
└────────────────────────────────────────────────────┘
  ↑                                                  ↓
┌────────────────────────────────────────────────────┐
│ Layer 3: ML OPERATIONS (MLOps)                     │
│ ├─ Model Registry (versioning, metadata)           │
│ ├─ Training Pipeline (distributed training)        │
│ ├─ Evaluation Framework (metrics, validation)      │
│ └─ Monitoring (model performance, drift)           │
└────────────────────────────────────────────────────┘
  ↑                                                  ↓
┌────────────────────────────────────────────────────┐
│ Layer 2: DATA INFRASTRUCTURE                       │
│ ├─ Data Lake (centralized storage)                 │
│ ├─ Data Pipeline (collection, cleaning, labeling)  │
│ ├─ Analytics (dashboards, real-time insights)      │
│ └─ Logging (all events, decisions, failures)       │
└────────────────────────────────────────────────────┘
  ↑                                                  ↓
┌────────────────────────────────────────────────────┐
│ Layer 1: ROBOT FLEET                               │
│ ├─ Edge Inference (on-robot GPU)                   │
│ ├─ Data Collection (local buffering)               │
│ ├─ Health Monitoring (onboard diagnostics)         │
│ └─ Failsafe Operation (local policies)             │
└────────────────────────────────────────────────────┘
```

---

## Layer 1: Robot Fleet

### Individual Robot

```
┌─────────────────────────────────┐
│         ROBOT NODE              │
├─────────────────────────────────┤
│                                 │
│  Perception:                    │
│  ├─ RGB camera (1080p, 30fps)   │
│  ├─ Depth sensor                │
│  └─ Joint encoders              │
│                                 │
│  Inference:                     │
│  ├─ Edge GPU (RTX 2060)         │
│  ├─ Current policy (50M params) │
│  └─ Latency <100ms              │
│                                 │
│  Data Collection:               │
│  ├─ Video buffer (local SSD)    │
│  ├─ Action log                  │
│  ├─ Success/failure records     │
│  └─ Compressed, ~5GB/day        │
│                                 │
│  Health Monitoring:             │
│  ├─ CPU/GPU utilization        │
│  ├─ Disk space remaining        │
│  ├─ Network connectivity        │
│  ├─ Motor temperature           │
│  └─ Gripper pressure            │
│                                 │
│  Failsafe:                      │
│  ├─ Local baseline policy       │
│  ├─ Collision detection         │
│  ├─ Emergency stop button       │
│  └─ Network fallback mode       │
│                                 │
└─────────────────────────────────┘
```

**Responsibilities**:
- Run inference 100+ times per second
- Collect data continuously
- Report health status every 30 seconds
- Detect and report failures
- Operate offline if network fails

---

## Layer 2: Data Infrastructure

### Data Flow

```
┌─ Robot 1 (daily collection)
│  └─ Video, actions, results → Local SSD (5GB)
│
├─ Robot 2-100 (same)
│  └─ Video, actions, results → Local SSD (5GB each)
│
├─ Weekly Sync (Sunday night)
│  └─ All robots upload to Central Data Lake
│      └─ 500GB compressed (from 500GB raw)
│
├─ Data Lake (Cloud/S3)
│  ├─ Raw videos (3TB/month)
│  ├─ Processed data (300GB/month)
│  ├─ Labels/annotations (50GB/month)
│  └─ Metadata (task IDs, success, time, robot)
│
├─ Analytics
│  ├─ Success rate by task/robot
│  ├─ Failure categorization
│  ├─ Bottleneck identification
│  └─ Performance trends
│
└─ Logging System
   ├─ Every action taken
   ├─ Every error
   ├─ Every model update
   └─ Every deployment
```

### Data Quality Pipeline

```
Raw Data → Quality Check → Cleaning → Labeling → Storage

Quality checks:
  ✓ Video not corrupted (filesize, frame count)
  ✓ Action is valid (within workspace)
  ✓ Success label is reasonable
  ✓ Timestamps are sequential
  ✓ Robot was healthy during collection

Cleaning:
  ✓ Remove videos with motion blur
  ✓ Remove outlier actions
  ✓ Fix mislabeled success/failure
  ✓ De-duplicate similar frames

Labeling:
  ✓ Human review of 1% of trials
  ✓ Automatic labeling of 99% (classifier)
  ✓ Edge cases sent to expert
```

---

## Layer 3: ML Operations (MLOps)

### Model Lifecycle

```
┌─────────────────────────────┐
│  Training Data (700K/week)  │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│ Model v24 Training          │
│ ├─ Architecture (from v23)  │
│ ├─ Hyperparameters (locked) │
│ ├─ Data split (train/val)   │
│ └─ 24 hours on GPU cluster  │
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│ Evaluation & Validation     │
│ ├─ Metrics on test set      │
│ ├─ Compare to v23           │
│ ├─ Automated checks:        │
│ │  - Not worse overall      │
│ │  - No catastrophic drops  │
│ │  - Confidence interval ok  │
│ └─ Manual review (5 min)    │
└─────────────┬───────────────┘
              ↓
        ┌─────┴─────┐
        ↓           ↓
    PASS        FAIL
    ↓           ↓
  Deploy    Investigate
   v24      (fix training)
   ↓           ↓
 A/B test   Try v25
```

### Model Registry

```
Model v1:  Baseline (85% accuracy)
  └─ weights.pt (500MB)
  └─ hyperparameters.yaml
  └─ metadata: trained 2024-01-01, 150 tasks
  └─ status: archived

Model v23: Current production (92% accuracy)
  └─ weights.pt (500MB)
  └─ hyperparameters.yaml
  └─ training_history (loss curve)
  └─ test_metrics (per-task accuracy)
  └─ status: production (100 robots)

Model v24: Candidate (92.1% accuracy)
  └─ weights.pt
  └─ hyperparameters.yaml
  └─ status: candidate (A/B test on 10 robots)

Model v25: In training (ETA 2 hours)
  └─ status: training (GPU cluster)
```

---

## Layer 4: Orchestration & Control

### Task Scheduler

```
Incoming Tasks Queue:
┌────────────────────────┐
│ Pick red ball (UrgentA)│ ← From warehouse system
│ Place on shelf (Normal)│
│ Fetch invoice (Normal) │
│ Clean spillage (High)  │
│ Package box (Normal)   │
│ ...                    │
└────────────────────────┘
        ↓
  Prioritize (UrgentA > High > Normal)
        ↓
  Allocate to Robots:
    Robot 1: Idle → Pick red ball
    Robot 2: Busy → Waiting
    Robot 3: Idle → Place on shelf
    ...
        ↓
  Execute in parallel
        ↓
  Queue next batch
```

### Fleet Manager

```
Monitors 100 robots in real-time:

  ✓ 95 healthy and working
  ✓ 3 temporarily blocked (waiting for human)
  ✓ 1 physically stuck (needs intervention)
  ✓ 1 offline (network issue)

Actions:
  - Reassign tasks from stuck robot to others
  - Alert maintenance for physical robot
  - Retry network connection to offline robot
  - Continue operations at 99% capacity
```

### Safety Monitor

```
Before executing action, check:
  ✓ Is workspace clear? (no humans detected)
  ✓ Is action within robot limits? (joint bounds)
  ✓ Would gripper collision with table?
  ✓ Is robot healthy? (temperature, pressure ok)
  ✓ Is action approved by safety model?

All checks pass → Execute
Any check fails → Reject action, log incident
```

---

## Layer 5: Human Interface

### Operator Dashboard

Real-time view of entire fleet:

```
FLEET STATUS (RIGHT NOW)
├─ Overall Success Rate: 91.2% (98% target: ✓ GREEN)
├─ Healthy Robots: 98/100 (98% target: ✓ GREEN)
├─ Tasks Completed Today: 12,450
├─ Average Latency: 145ms (< 200ms target: ✓ GREEN)
├─ Model Version: v23 (92% accuracy)
└─ Last Update: 2 seconds ago

ALERTS (REQUIRES ATTENTION)
├─ ⚠️ Robot 47 stuck (5 minutes) → Recommend reset
├─ ⚠️ Model v24 test failed → Investigate training
└─ ⚠️ Network latency high (450ms) → Check WiFi

PERFORMANCE TRENDS (7 DAYS)
├─ Success: 88% → 91% (↑ 3% improvement ✓)
├─ Latency: 200ms → 145ms (↓ faster ✓)
└─ Healthy: 92% → 98% (↑ more reliable ✓)

TOP FAILURE MODES (TODAY)
├─ 25% Gripper slip (mechanical wear)
├─ 15% Occlusion (camera angle)
├─ 12% Reach error (IK solver)
└─ 48% Other (need investigation)
```

### Incident Response

```
Incident Detected → Alert → Investigation → Resolution

Example: Success rate drops to 80%
  1. Alert (1 min): "Success rate below threshold"
  2. Investigation (5 min):
     - Check which tasks failed
     - Check if model or hardware issue
     - Check if external conditions changed
  3. Decision:
     - If model: Rollback to v23
     - If hardware: Reboot affected robots
     - If environment: Adjust safety bounds
  4. Action (2 min): Execute fix
  5. Verify (5 min): Confirm success rate recovered
  6. Document: Record incident in incident log
```

---

## Integration with Factory Systems

### APIs & Data Feeds

```
Robot Fleet ←→ Factory Systems

Feeds FROM factory:
  GET /task-queue?facility=warehouse-1
    ← List of tasks to execute

  GET /item-location?item_id=SKU-123456
    ← Where is this item physically?

  GET /safety-bounds?zone=A1
    ← What are movement restrictions?

Feeds TO factory:
  POST /task-completion
    ← Task done, item moved to X,Y

  POST /robot-failure
    ← Robot stuck, needs human help

  POST /inventory-update
    ← Updated inventory based on completed tasks
```

### Constraints & Boundaries

```
Physical Constraints:
  - Each facility has different layout
  - Moving obstacles (other robots, humans)
  - Different lighting conditions
  - Temperature variations

Operational Constraints:
  - Prioritize urgent tasks
  - Avoid interfering with humans
  - Don't start new task if shutting down soon
  - Balance load across all robots

Safety Constraints:
  - Humans always have right-of-way
  - No operation in restricted zones
  - Emergency stop button always works
  - Never exceed joint limits
```

---

## System Properties

### Reliability

Target: 99.9% uptime (22 hours downtime per year)

```
Single robot: 99.5% uptime (4.3 hours downtime/year)
100 robots: 99% uptime (87.6 hours downtime total)
  But distributed: Likely 1-2 robots down at any time

Redundancy strategies:
  - Hot-backup models on each robot
  - Network failover to 4G modem
  - Local policy fallback
  - Task requeue to other robots
```

### Scalability

Growing from 10 to 1000 robots:

```
10 robots:     Simple (one server sufficient)
100 robots:    Moderate (need monitoring, logging)
1000 robots:   Complex (distributed training, federation)

Bottlenecks by scale:
  10→100:  Network bandwidth (solution: compress data)
  100→1000: Training compute (solution: distributed training)
  1000+:   Operational complexity (solution: automation)
```

### Latency

Real-time requirements:

```
Perception:      30ms (camera to detection)
Planning:        50ms (scene understanding)
Decision:        10ms (which action to take)
Execution:       30ms (send commands to motors)
Total:          ~120ms (< 200ms requirement ✓)

P95 latency:     < 250ms (network variance)
P99 latency:     < 500ms (acceptable edge cases)
```

---

## Key Design Principles

✅ **Separation of Concerns**: Each layer is independent (can upgrade without touching others)

✅ **Observability**: Every decision is logged and traceable

✅ **Graceful Degradation**: System operates at reduced capacity, not fails completely

✅ **Automation**: Humans only for decisions, not routine monitoring

✅ **Safety First**: No way to accidentally hurt someone

✅ **Cost Efficiency**: Pay for what you use (no over-provisioning)

✅ **Extensibility**: Easy to add new robots, tasks, factories

---

## Next: Deployment Strategies

With this architecture in place, how do you deploy models safely?

[→ Continue to Section 2: Deployment Strategies](deployment-strategies.md)

