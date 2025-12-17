# The Learning Spectrum: BC vs Diffusion vs RL

Visualization of the complete learning paradigm spectrum for embodied AI.

---

## Overview: Three Paradigms, One Goal

```
Goal: Learn policy π(a|s) that maps observations → actions

                    BC (Imitation)
                    ↓
            Behavioral Cloning
            ↓              ↑
    (supervised)      (one-shot)
            ↓              ↑
    Copy expert    Multimodal actions
    100 demos       200 demos
    75% success     85% success
    Hours to deploy Days to deploy


                    Diffusion
                    ↓
            Trajectory Generation
            ↓              ↑
    (iterative)     (conditional)
            ↓              ↑
    Denoise from    Smooth outputs
    noise via        Handles multiple
    diffusion        solutions
    200 demos        250ms latency
    85% success      Good generalization


                    RL (Trial-and-Error)
                    ↓
            Policy Gradient + Critic
            ↓              ↑
    (exploration)    (exploitation)
            ↓              ↑
    Learn from      Superhuman
    interactions     performance
    10K trials       95% success
    Weeks to train   Excellent generalization
```

---

## Comparison Axes

### Axis 1: Data Efficiency vs Exploration

```
Data Efficiency (utilizes demonstrations well)
            ← BC ← Diffusion ← RL →
                                Exploration (learns from mistakes)

BC:       Uses every demonstration, high variance in small batches
Diffusion: Tolerates noisy data, bootstraps new trajectories
RL:       Ignores demos, explores environment, slow but thorough
```

### Axis 2: Speed vs Quality

```
Speed (inference latency)
            ← BC (50ms)
                    ← RL (100ms)
                            ← Diffusion (200-500ms) →
                                Quality (success rate)

BC:       Fast, but lower success (60-80%)
RL:       Moderate speed, higher success (85-95%)
Diffusion: Slower, good success (80-90%), but variable quality
```

### Axis 3: Sample Efficiency vs Generalization

```
Sample Efficiency (data needed per performance point)
            ← BC (1 demo per 1% improvement)
                    ← Diffusion (2 demos per 1%)
                            ← RL (100 trials per 1%) →
                                Generalization (novel objects)

BC:       Efficient with data, poor generalization (30-50%)
Diffusion: Balanced efficiency, good generalization (60-75%)
RL:       Inefficient with data, best generalization (75-85%)
```

---

## Decision Matrix: Which Method?

```
                  BC          Diffusion        RL
                  ─────────────────────────────────────
Data?             ✓ Yes       ✓ Yes            ✗ No
(<100 demos)

Quality?          ⚠ Medium    ✓ Good           ✓✓ Great
(>80% needed)

Speed?            ✓✓ Fast     ⚠ Moderate       ✓ Fast
(<100ms)

Multimodal?       ✗ No        ✓ Yes            ✓ Yes
(many solutions)

Exploration?      ✗ No        ⚠ Limited        ✓ Yes
(unknown optimum)

Deployment?       ✓ Hours     ⚠ Days           ✗ Weeks
(time to ready)

Safety?           ✓✓ Safe     ✓ Safe           ⚠ Risky
(errors compound) (smooth)     (may crash during training)

Recommendation:
├─ Have demos, need fast → BC
├─ Have demos, need quality → Diffusion or BC+RL
├─ Need superhuman, have time → RL
└─ Unsure → Start with BC, add Diffusion for challenging cases
```

---

## Performance Landscape

```
                   Success Rate (%)
                   100 ├──────────────────────────────────
                       │    RL (after 2 weeks)
                    95 │   ╭────────────
                       │  ╱
                    90 │ ╱  Diffusion (after 2 days)
                       │╱╭──────────
                    85 │ │
                       │ │  BC (after 8 hours)
                    80 │ │╭───
                       │ ││
                    75 │ ││  BC+RL (best combo)
                       │ ││╭──────
                    70 │ │││
                       │ │││
                    65 │ │││  DAgger (BC + corrections)
                       │ │││╭
                    60 │ ││││
                       │ ││││
                    55 │ ││││
                       │ ││││  BC baseline
                    50 │─││││
                       │0 2 4  6  8  10 12 14 16 18 20
                       │Days of development
```

**Interpretation**:
- BC: Fast to 50-60% (1-2 days), plateau at 75%
- DAgger: Gradual improvement with expert corrections
- Diffusion: Better quality from fewer demos, takes longer to code
- RL: Slow start (week 1 is data collection), then rapid improvement
- BC+RL: Best balance (start with BC, fine-tune with RL)

---

## Data-Performance Tradeoff

```
Success Rate (%)
         100 ├──────────────────────────────────
             │
          95 │      RL (curves upward)
             │     ╱╱
          90 │    ╱╱ Diffusion (smooth growth)
             │   ╱╱╭─────
          85 │  ╱╱  BC+RL
             │ ╱╱ ╱──
          80 │╱ ╱╱
             │╱╱  BC (plateaus)
          75 │ ╱
             │╱
          70 └─────────────────────────────────
             0   200   500   1000  2000  5000  10000
             Number of Demonstrations/Trials

BC:        Improves with more demos up to 500, then plateaus at 75-80%
Diffusion: Steady improvement, reaches 85% at 300 demos, 90% at 1000
RL:        Slow start (needs 1000 trials to beat BC), then outpaces
BC+RL:     BC gives initial boost, RL fine-tuning adds final 10%

Key insight: For 300 demonstrations, Diffusion is best
             For 10K trials available, RL is best
             For <100 demos, BC is only option
```

---

## Task Suitability

```
                    BC      Diffusion    RL
─────────────────────────────────────────────────

Reaching         ✓✓✓       ✓✓✓         ✓✓✓
(simple, clear)

Grasping         ✓✓        ✓✓✓         ✓✓✓
(multimodal)

Pushing          ✓         ✓✓✓         ✓✓
(dynamic)

Insertion        ✓         ✓✓✓         ✓✓
(precise)

Long horizon     ✗         ✓           ✓✓✓
(many steps)

Navigation       ⚠         ✓✓          ✓✓✓
(sparse rewards)

Dexterous       ✗          ⚠           ✓✓✓
(complex)

Novel objects    ✗         ✓✓          ✓✓✓
(generalization)

Legend: ✓✓✓ Excellent, ✓✓ Good, ✓ Fair, ⚠ Challenging, ✗ Poor
```

---

## Time Investment

```
BC:
  Development:   6-8 hours
  ├─ Data collection: 2-3 hours
  ├─ Network design: 2 hours
  ├─ Training: 1 hour
  └─ Testing: 1 hour

Diffusion:
  Development:   2-3 days
  ├─ Data collection: 2-3 hours (same as BC)
  ├─ Model implementation: 4-6 hours
  ├─ Training: 4-8 hours
  ├─ Hyperparameter tuning: 4-8 hours
  └─ Testing and debugging: 2-4 hours

RL:
  Development:   1-4 weeks
  ├─ Environment setup: 2-3 days
  ├─ Reward engineering: 3-5 days
  ├─ Algorithm implementation: 2-3 days
  ├─ Training: 5-10 days
  ├─ Sim-to-real transfer: 3-5 days
  └─ Real robot validation: 2-3 days
```

---

## Feature Comparison Table

```
Feature              BC           Diffusion      RL
─────────────────────────────────────────────────────────
Demos needed         100-500      200-500        0
Training time        1-4 hours    4-16 hours     1-4 weeks
Inference latency    50ms         200-500ms      100ms
Success rate         60-80%       80-90%         90-98%
Novel object gen.    30-50%       60-75%         75-85%
Smooth outputs       No           Yes            Variable
Handles multimodal   No           Yes            Yes
Exploration          No           Limited        Extensive
Requires reward      No           No             Yes
Safety training      High         High           Low
Debugging            Easy         Medium         Hard
Real robot ready     <1 day       2-3 days       1-2 weeks
Code maturity        Stable       Emerging       Mature
```

---

## Workflow Recommendation

### Scenario 1: Startup (No Time, Limited Data)

```
Week 1: Collect 200 grasping demonstrations (expert)

Week 2: Implement BC
  Day 1: Data loading, preprocessing (4 hours)
  Day 2: CNN architecture, training (4 hours)
  Day 3: Testing on robot (4 hours)

Result: 75% grasping success in 3 days
Trade-off: Can't improve further without more demos/expert corrections
```

### Scenario 2: Research (Time Available, Want Best Results)

```
Weeks 1-2: Collect 500 diverse demonstrations

Weeks 3-4: Implement Diffusion Policy
  - More complex but better performance
  - 85-90% success on novel objects
  - 2-3 weeks to production ready

Week 5: Fine-tune with RL if needed
  - Start from diffusion baseline
  - 2-3 days RL training
  - Reach 92%+ success
```

### Scenario 3: Production Robotics (Performance Critical)

```
Month 1: Collect 1000+ demonstrations across variations

Month 2-3: Train with BC+RL hybrid
  - Warm-start network with BC (1 week)
  - Fine-tune with RL (2 weeks)
  - Validate on real robot (1 week)

Result: 95%+ reliable performance
Safety: Multiple fallback policies
Deployment: 3-4 months to production ready
```

---

## Summary Grid

```
                Quick Win      Balanced         Optimal
                ─────────────────────────────────────────
Best For:       MVP/demo      Research         Production
Time:           Days          Weeks            Months
Data:           100-200       500-1000         5000+
Performance:    70-75%        80-90%           95%+
Method:         BC            Diffusion        BC+RL
Safety:         Manual        Checkpoints      Redundancy
```

---

## Key Insights

✅ **No silver bullet** - Each method has tradeoffs
✅ **Combination is powerful** - BC startup, RL fine-tuning beats all individual
✅ **Data unlocks performance** - More demonstrations → better results
✅ **Deployment timeline matters** - BC in days, RL in months
✅ **Task determines method** - Simple task? BC. Complex? RL.

---

## Further Exploration

- **Choose BC if**: You want a working policy in days with limited budget
- **Choose Diffusion if**: You want good generalization with medium effort
- **Choose RL if**: You have time and want best possible performance
- **Choose BC+RL if**: You want both speed to baseline and high final performance

---

**Next:** [Training Pipeline & Implementation →](training-pipeline.md)

