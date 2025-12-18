# Fleet Architecture: System Design at Scale

Complete system design for 100+ robots learning together.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     GLOBAL LEARNING SERVER                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │  Model Storage │  │ Data Pipeline  │  │ Model Trainer  │ │
│  │  (150 tasks)   │  │  (data mgmt)   │  │  (GPU cluster) │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │  Monitoring    │  │  Analytics     │  │  Experimentation
│  │  Dashboard     │  │  (metrics)     │  │  (A/B testing) │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└──────────────────────────────────────────────────────────────┘
              ↑                                    ↓
              │                                    │
    ┌─────────┼────────────┬──────────┬───────────┼──────────┐
    │         │            │          │           │          │
  [Robot 1] [Robot 2] [Robot 3] ... [Robot 50] [Robot 100]
    │         │            │          │           │          │
    ├─ Local  ├─ Local     ├─ Local   ├─ Local   ├─ Local   │
    │ storage │ storage    │ storage  │ storage  │ storage  │
    │         │            │          │          │          │
    ├─ Collect├─ Collect   ├─ Collect ├─ Collect ├─ Collect │
    │ data    │ data       │ data     │ data     │ data     │
    │ (1000/day)           │ (1200/day)         │ (800/day) │
    │         │            │          │          │          │
    └─────────┼────────────┴──────────┴──────────┴──────────┘
              │
    Weekly Model Sync + Aggregation
```

---

## Component Breakdown

### 1. Global Learning Server

```
┌─────────────────────────────────────┐
│    Central Learning Infrastructure  │
├─────────────────────────────────────┤
│                                     │
│  Model Storage:                     │
│  ├─ Current weights (150 tasks)     │
│  ├─ Historical versions             │
│  ├─ Rollback capability             │
│  └─ A/B testing versions            │
│                                     │
│  Data Pipeline:                     │
│  ├─ Receive raw data from robots    │
│  │  (videos, actions, results)      │
│  ├─ Data quality checks             │
│  ├─ Labeling pipeline               │
│  └─ Store in data lake              │
│                                     │
│  Training Infrastructure:           │
│  ├─ 32-GPU cluster (distributed)    │
│  ├─ Multi-task training scheduler   │
│  ├─ Distributed data parallelism    │
│  └─ Federated learning aggregation  │
│                                     │
│  Monitoring:                        │
│  ├─ Fleet health dashboard          │
│  ├─ Model performance tracking      │
│  ├─ Anomaly detection               │
│  └─ Alerts for failures             │
│                                     │
│  Analytics:                         │
│  ├─ Success rates by task/robot     │
│  ├─ Failure analysis (automated)    │
│  ├─ Cost tracking                   │
│  └─ ROI calculation                 │
│                                     │
│  Experimentation:                   │
│  ├─ A/B testing framework           │
│  ├─ New algorithm testing           │
│  ├─ Gradual rollout                 │
│  └─ Rollback on failure             │
│                                     │
└─────────────────────────────────────┘
```

### 2. Individual Robot

```
┌──────────────────────────────────┐
│       Individual Robot Node       │
├──────────────────────────────────┤
│                                  │
│  Perception:                     │
│  ├─ RGB camera (1080p, 30fps)    │
│  ├─ Depth sensor                 │
│  ├─ Joint encoders               │
│  └─ Force/torque sensor          │
│                                  │
│  Onboard Compute:                │
│  ├─ Edge GPU (RTX 2060)          │
│  ├─ Inference runtime            │
│  ├─ Policy (~50M params)         │
│  └─ Fallback policy              │
│                                  │
│  Control:                        │
│  ├─ Real-time control loop       │
│  │  (100 Hz)                     │
│  ├─ Safety constraints           │
│  ├─ Collision detection          │
│  └─ Graceful degradation         │
│                                  │
│  Data Collection:                │
│  ├─ Record videos (raw)          │
│  ├─ Log actions taken            │
│  ├─ Record results (success/fail)│
│  └─ Compress + upload daily      │
│                                  │
│  Local Learning:                 │
│  ├─ Download global model weekly │
│  ├─ Fine-tune on local data      │
│  │  (optional, if needed)        │
│  ├─ A/B test new models          │
│  └─ Report performance back      │
│                                  │
└──────────────────────────────────┘
```

---

## Data Flow: Continuous Learning Loop

```
DAY 1:
  Robot 1 → Collect 1000 trials
  Robot 2 → Collect 1000 trials
  ...
  Robot 100 → Collect 1000 trials
           ↓
         Total: 100K new data points


WEEKLY SYNC (Sunday):
  All robots → Upload data
           ↓
  Central server: Aggregate 700K data points


  Training phase (24 hours):
  ├─ Process data (quality checks)
  ├─ Train on new + historical data
  ├─ Multi-task learning across 150 tasks
  ├─ Distributed training on 32 GPUs
  ├─ Evaluate on validation set
  └─ A/B test on subset of robots


  Distribution phase (Monday morning):
  ├─ Download new weights to all robots
  ├─ Run safety checks (not worse than before)
  ├─ Deploy to production robots
  └─ Monitor first day for issues


WEEK 2-4:
  Repeat: Collect → Sync → Train → Deploy


RESULTS:
  Week 1:  Fleet average 78% success
  Week 4:  Fleet average 80% success (+2%)
  Month 2: Fleet average 81% (+1%, diminishing returns)
  Month 6: Fleet average 82% (plateau reached)
```

---

## Communication Topology

```
HIGH BANDWIDTH (Initial):
  ├─ Robot → Server: Upload 100GB/week per robot
  │  (compressed video, raw actions, results)
  │
  └─ Server → Robot: Download 50MB/week per robot
     (new model weights, task definitions)

OPTIMIZED (With compression):
  ├─ Robot → Server: 10GB/week per robot
  │  (delta updates, compressed frames)
  │
  └─ Server → Robot: 10MB/week per robot
     (compressed model, quantized weights)

NETWORK REQUIREMENTS:
  100 robots × 10GB/week ÷ 7 days = 140GB/day
  = 1.3 Gbps average bandwidth needed

  Typical facility:
  ├─ Wired (office): 10 Gbps available ✓
  ├─ WiFi (warehouse): 2-5 Gbps shared ✓
  └─ 4G fallback: 100 Mbps (slow, but works)

HANDSHAKE:
  Monday 8am: Robot connects
  ├─ Authenticate (30 sec)
  ├─ Check last known version (5 sec)
  ├─ Download model if needed (30 sec)
  └─ Resume data collection (ready to go)
```

---

## Fault Tolerance

```
Scenario 1: Single Robot Fails
  ├─ Motor broken
  ├─ Network down
  └─ GPU crashes

  System response:
  ├─ Detect no heartbeat for 5 minutes
  ├─ Alert operator
  ├─ Continue training on 99 other robots
  ├─ Fleet average barely affected (1%)
  └─ When fixed, robot syncs and catches up


Scenario 2: Network Outage (4 hours)
  ├─ All robots lose connectivity
  ├─ Local policies continue running
  ├─ Robots buffer data locally
  ├─ When network recovers:
  │  ├─ Upload buffered data
  │  ├─ Server processes as if delayed
  │  └─ Training continues
  │
  └─ No data loss, slight training delay


Scenario 3: Server Fails
  ├─ Central server crashes
  ├─ All robots detect missing weekly sync
  ├─ Robots continue with current model
  ├─ If outage > 2 weeks:
  │  ├─ Robots become stale
  │  └─ Fall back to baseline policy
  │
  ├─ When fixed:
  │  ├─ Replay all buffered data
  │  ├─ Retrain all models
  │  └─ Redeploy
  │
  └─ Data durability: 100% (replicated storage)


Scenario 4: Model Corruption
  ├─ Bad weights deployed
  ├─ Success rate drops to 50%
  ├─ Monitoring system detects anomaly
  ├─ Automatic rollback to previous version
  └─ Incident resolved in <1 hour
```

---

## Deployment Strategy

```
BLUE-GREEN DEPLOYMENT:

Blue Fleet (Current):
  ├─ 50 robots running production model
  ├─ Success rate: 80%
  ├─ Processing 2500 items/day

Green Fleet (New):
  ├─ 50 robots running new model
  ├─ Success rate: 80.5% (measured)
  ├─ Processing 2500 items/day

If Green is better:
  ├─ Swap: Green becomes new Blue
  ├─ Deploy new model to all 100
  └─ Benefit: +0.5% success = +1250 items/day!

If Green is worse:
  ├─ Keep current model
  ├─ Investigate why (failure analysis)
  └─ Try different hyperparameters
```

---

## Operations Team Structure

```
Central Learning Server Team (8 people):
  ├─ ML Engineer (1): Training pipeline, model development
  ├─ Data Engineer (2): Data quality, pipeline management
  ├─ DevOps (1): Infrastructure, monitoring, reliability
  ├─ Product Manager (1): Prioritizing new tasks
  ├─ Operations (2): Fleet coordination, issue resolution
  └─ Intern (1): Data labeling, annotation

Facility Teams (10 facilities × 2 people):
  ├─ Roboticist (1): Maintenance, troubleshooting
  └─ Operations (1): Safety, task management
```

---

## Success Metrics

```
Tracked Daily:
  ├─ Fleet success rate (target: >80%)
  ├─ Uptime (target: >95%)
  ├─ Mean latency (target: <200ms)
  ├─ Tasks operational (target: 150+)
  └─ Cost per item (target: $4-6)

Tracked Weekly:
  ├─ Model performance improvement
  ├─ New data collected (700K trials/week)
  ├─ Failed rollouts (should be 0)
  └─ Critical incidents

Tracked Monthly:
  ├─ New tasks added
  ├─ Cost reduction progress
  ├─ Generalization improvement
  └─ Team velocity
```

---

## Capacity Planning

```
Current (100 robots):
  ├─ Data collection: 100K trials/day
  ├─ Training data: 3M trials/month
  ├─ Inference: 100K predictions/day
  └─ Cost: $8M/year

Projected Growth (in 3 years):
  ├─ Robots: 500
  ├─ Tasks: 500+
  ├─ Data collection: 500K trials/day
  ├─ Training data: 15M trials/month
  ├─ Inference: 500K predictions/day
  └─ Cost: $30M/year (but revenue grows faster)

Headroom:
  ├─ GPU cluster: 5× expansion capacity
  ├─ Data storage: 10× expansion planned
  ├─ Network: 4× current bandwidth
  └─ Should handle 3-5× growth before major upgrades
```

---

## Key Design Principles

✅ **Redundancy**: No single point of failure
✅ **Scalability**: Add 10 robots without redesign
✅ **Monitoring**: Visibility into all systems
✅ **Safety**: Graceful degradation, not catastrophic failure
✅ **Automation**: Minimal manual intervention
✅ **Testing**: A/B testing before global deployment

---

**Previous Section:** [Scaling Pipeline →](scaling-pipeline.md)

