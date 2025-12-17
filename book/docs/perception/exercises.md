# Module 3: Perception & Sim-to-Real Transfer Exercises

These exercises guide you through applying sensor fusion, sim-to-real transfer, and domain randomization to real robotics challenges. Work through them in order, and compare your solutions with the provided solutions.

---

## Exercise 1: Identify Sim-to-Real Gaps in a Task

### Objective

Given a simulated robotic grasping task, identify the four major sim-to-real gaps and propose specific mitigations.

### Scenario

You've trained a policy in Gazebo for a humanoid to grasp bottles from a table. The policy achieves 98% success in simulation but fails 60% of the time on a real robot. Your job is to diagnose why.

### Task

For each of the four sim-to-real gaps (from Module 3 intro), describe:

1. **What's missing from the simulation** that causes the real robot to fail
2. **Why it matters** (impact on task success)
3. **How to mitigate it** (specific changes to simulation or policy)

Complete the analysis below:

**Gap 1: Sensor Fidelity**

What's missing from simulation?
-

Why it matters for grasping?
-

Mitigation strategy?
-

**Gap 2: Physics Simulation**

What's missing from simulation?
-

Why it matters for grasping?
-

Mitigation strategy?
-

**Gap 3: Environmental Variation**

What's missing from simulation?
-

Why it matters for grasping?
-

Mitigation strategy?
-

**Gap 4: Timing & Synchronization**

What's missing from simulation?
-

Why it matters for grasping?
-

Mitigation strategy?
-

### Success Criteria

✅ You've identified specific, realistic differences (not vague "stuff is different")
✅ For each gap, explanation includes why it causes real failures
✅ Mitigation strategies are concrete (e.g., "add motor delay [0-50ms]" not "improve physics")

### Solution

**Gap 1: Sensor Fidelity**

What's missing: Gazebo renders perfect RGB images. Real cameras have rolling shutter (frame rows captured at different times), motion blur (moving objects smeared), and variable exposure (auto-adjusts to lighting). The gripper moves at 0.5 m/s, causing ~8mm of motion blur across the 640-pixel image. CNN trained on sharp images has never seen blurred grippers.

Why it matters: Vision-based grasp detection relies on identifying gripper position. If gripper appears blurred or distorted, the CNN's confidence drops from 0.95 (sim) to 0.50 (real), causing wrong grasp points. Result: object drops.

Mitigation:
- Add camera noise plugin to Gazebo (Gaussian σ=0.03-0.05)
- Enable rolling shutter simulation
- Train perception model with motion blur augmentation
- Fine-tune on 100 real camera images before deployment

**Gap 2: Physics Simulation**

What's missing: In Gazebo, joint commands execute instantly. Real motors have inertia, acceleration limits, and saturation. A motor with 100 Nm max torque and 0.5 kg⋅m² inertia reaches 1 rad/s in ~5ms, not instantly. Additionally, friction is modeled as constant μ in Gazebo. Real friction depends on velocity and surface condition (wet bottle: μ=0.2 vs dry: μ=0.7).

Why it matters: Gripper force tuned in sim (50 N) assumes instant response and constant friction. On real hardware, motor is slower, and wet bottles slip at 30 N instead of expected 50 N. Result: gripper force insufficient, object drops.

Mitigation:
- Add motor dynamics to simulation (acceleration limits, max torque)
- Randomize friction coefficient during training [0.3, 1.0]
- Measure real motor response curves, inject into sim
- Implement force feedback control on real robot (not just open-loop gripper force)

**Gap 3: Environmental Variation**

What's missing: Gazebo uses fixed bottle position (0.5, 0.1, 0.9), fixed friction (μ=0.7), fixed lighting (sun from north). Real world has placement tolerance (±2cm), material variation (glass bottle=0.4, plastic=0.6, wet=0.2), and lighting that changes with time of day, weather, and indoor/outdoor conditions.

Why it matters: Policy learned to grasp bottles at exact position. Real placement varies ±2cm, causing gripper approach angle mismatch. Gripper force tuned for μ=0.7 fails on wet surfaces (μ=0.2). Vision system trained on one lighting condition fails under LED lights or shadows.

Mitigation:
- Randomize object position ±5cm in training
- Randomize object friction μ ∈ [0.3, 1.0] covering all materials
- Randomize lighting (direction, intensity) in simulation
- Randomize object appearance (colors, textures)

**Gap 4: Timing & Synchronization**

What's missing: Gazebo simulation is deterministic with synchronized components. Perception runs at exactly 30 Hz with zero jitter. Real systems are asynchronous—camera frames arrive at variable times, network packets are occasionally lost, CPU load varies. A frame arrives at 20ms, then the next at 25ms (not 20ms again). Perception sometimes takes 50ms instead of 30ms.

Why it matters: Policy trained on perfect timing assumes perception is always available and fresh. Real jitter causes delayed feedback, making robot respond to stale information. Lost camera frame leaves policy with no perception input, causing blind movement.

Mitigation:
- Add simulated latency to perception (uniform [20, 50]ms)
- Simulate dropped frames (random 5% packet loss)
- Retrain policy to tolerate timing variations
- Use Kalman filter on real robot to smooth out jitter

---

## Exercise 2: Design Domain Randomization Strategy

### Objective

Design a complete domain randomization strategy for vision-based grasping, including parameter ranges and rationale.

### Scenario

You're training a gripper to grasp unknown objects in a warehouse. Objects vary in size, material, color. Lighting is mixed (natural daylight + fluorescent). You have 1 week to prepare sim training. You want >80% real-world success.

### Task

Fill in the domain randomization table below. For each parameter, specify:
- **Randomization Range**: What values to sample during training
- **Rationale**: Why this range (match real-world variation)
- **Impact**: How does this help sim-to-real transfer?

**Complete the table:**

| Parameter | Baseline Value | Randomization Range | Rationale | Impact |
|-----------|---|---|---|---|
| Object friction | 0.7 | [___,___] | | |
| Object size | radius=0.04m | [___,___]m | | |
| Object position | (0.5, 0.1) | ±___ m | | |
| Camera noise | σ=0 | σ ∈ [___,___] | | |
| Lighting intensity | 1.0 | [___,___] | | |
| Lighting direction | North (1,0,1) | Random from ___  | | |
| Motor delay | 0 ms | [___,___]ms | | |
| Table friction | 0.8 | [___,___] | | |

### Success Criteria

✅ Parameter ranges are realistic (not arbitrary extremes)
✅ Each range is justified with real-world variation reasoning
✅ Chosen ranges are sufficient to handle warehouse variation

### Solution

| Parameter | Baseline Value | Randomization Range | Rationale | Impact |
|-----------|---|---|---|---|
| Object friction | 0.7 | [0.3, 1.0] | Glass bottles (0.4) to rubber grips (0.9), plus dust/wetness variation | Gripper learns to adapt force; handles slippery & rough objects |
| Object size | radius=0.04m | [0.03, 0.08]m | Real objects vary ±100% in size (small cups to large jugs) | Approach angle generalizes; not overfit to one object size |
| Object position | (0.5, 0.1) | ±0.05 m | Manufacturing/placement tolerance ±2-5cm in warehouse bins | Spatial generalization; works from multiple positions |
| Camera noise | σ=0 | σ ∈ [0.02, 0.05] | Real cameras have quantization + Poisson noise; σ~2-5% typical | Vision CNN robust to real sensor noise |
| Lighting intensity | 1.0 | [0.5, 1.5] | Warehouse has LED (bright) + windows (variable) + shadows | Detection works across lighting conditions |
| Lighting direction | North (1,0,1) | Random unit vector | Sun angle varies (latitude, season, time); indoor lights omni-directional | Object detection invariant to lighting direction |
| Motor delay | 0 ms | [10, 100]ms | Real motor response + network latency: 5-50ms typical | Policy tolerant of command delays |
| Table friction | 0.8 | [0.4, 0.9] | Warehouse tables: clean (0.8) vs dusty/wet (0.4-0.5) | Stable grasp across table conditions |

**Key Insight**: Range reflects real variation, not extremes. You could randomize μ ∈ [0.1, 2.0] (extreme), but [0.3, 1.0] (realistic) is better for faster learning and better transfer.

---

## Exercise 3: Analyze Hardware-in-the-Loop Iteration

### Objective

Given real robot test data, plan the next hardware-in-the-loop (HiL) iteration to improve sim-to-real transfer.

### Scenario

Week 1-2: You trained a grasping policy in Gazebo with domain randomization. Achieved 90% success in sim.

Week 3: First hardware test on a real robot. Results:
- Trials: 50
- Successes: 30 (60% success rate)
- Failures analyzed:

| Failure Mode | Count | Example |
|---|---|---|
| Vision failure (wrong grasp point) | 12 | CNN misidentified bottle position in LED lighting |
| Gripper slip (object drops during hold) | 5 | Bottle friction lower than expected |
| Timing failure (robot moves incorrectly) | 3 | Late perception caused wrong reach position |

### Task

Based on this data, answer:

1. **Transfer Ratio**: Calculate real_success / sim_success. Is it acceptable?

2. **Primary Failure Mode**: Which gap is most impactful? How many trials failed due to this?

3. **Simulation Update**: For the top failure mode, what specific changes would you make to the Gazebo simulation?

4. **New Randomization**: How should you adjust randomization parameters for next week's training?

5. **Timeline**: What's your plan for Week 4-5?

### Success Criteria

✅ Transfer ratio calculated and assessed against 0.8 threshold
✅ Root cause identified with supporting data
✅ Simulation improvements are specific (not vague)
✅ New randomization ranges justified
✅ Plan is realistic (can execute in 1-2 weeks)

### Solution

**1. Transfer Ratio**
- Real success: 60%
- Sim success: 90%
- Transfer ratio: 60/90 = 0.67

Status: MARGINAL. Goal is ≥0.80, so you're below target. Good news: this ratio is salvageable with targeted sim improvements.

**2. Primary Failure Mode**

Failure counts:
- Vision: 12 trials (40% of failures)
- Gripper slip: 5 trials (17%)
- Timing: 3 trials (10%)

**Primary failure is vision** (40% of real failures). This aligns with Module 3 insight that sensor fidelity is the biggest sim-to-real gap for vision-based tasks.

**3. Simulation Update for Vision Failures**

Current assumption: Sim trains on perfect Gazebo renders
Real observation: CNN fails in LED lighting (auto-exposure changes image brightness dramatically)

Specific changes:
- **Add camera plugin**: Enable realistic sensor noise σ=0.03-0.05 (currently 0)
- **Simulate auto-exposure**: Vary image exposure by ±20% to simulate auto-gain
- **Add rolling shutter**: Rows of image captured at different times (currently instant render)
- **Randomize lighting more aggressively**: [0.3, 2.0] intensity (was [0.5, 1.5]), also randomize light color (blue daylight vs orange LED)

**4. New Randomization Parameters for Week 4**

Current ranges → New ranges:
- Camera noise: σ ∈ [0.02, 0.05] → σ ∈ [0.03, 0.10] (double uncertainty)
- Lighting intensity: [0.5, 1.5] → [0.3, 2.0] (wider range)
- Lighting direction: Random → Also randomize color temperature (daylight~6500K to tungsten~3000K)
- New: Motion blur simulation during gripper approach (add image blur based on velocity)

Gripper slip is secondary (only 5 trials). Could improve:
- Friction randomization [0.3, 1.0] → keep same (already covers wet bottles)
- But add velocity-dependent friction model (slip increases with speed)

**5. Week 4-5 Plan**

**Week 4:**
- Monday: Implement Gazebo camera plugin improvements
- Tuesday-Wednesday: Retrain policy with new randomization (1000 episodes, ~5 hours GPU)
- Thursday: Validate new policy in sim (should drop to 80% due to harder randomization)
- Friday: Prepare for testing

**Week 5:**
- Monday-Tuesday: Hardware test on real robot (50 more trials)
- Wednesday: Analyze new results
  - Expected outcome: Vision failures drop to ~5-7 (from 12)
  - Success rate expected: ~75% (60% → 75%)
  - Transfer ratio: 75/80 = 0.94 ✓ (GOOD!)
- Thursday-Friday: Fine-tune if needed

---

## Challenge Exercises (Optional)

### Challenge 1: Evaluate Randomization Effectiveness

Using `domain_randomization.py` from code examples:

1. Create two training runs:
   - Run A: No randomization (deterministic)
   - Run B: Full randomization (matching real variation)

2. For each run, collect:
   - Sim success rate (every 100 episodes)
   - Transfer ratio on 20 real robot trials

3. Compare: Does randomization hurt sim performance? Does it improve transfer?

**Expected result**: Run B has lower sim performance (~85% vs 98%) but much better transfer (~0.85 vs ~0.35).

### Challenge 2: Domain Adaptation with Fine-Tuning

Using `sim_to_real_evaluation.py`:

1. Train perception model on 10,000 synthetic images (Gazebo-rendered)
2. Test on 100 real images from robot camera
3. Fine-tune on subset (10, 20, 50 real images)
4. Plot: Accuracy vs number of real images used

**Expected result**: 10 images → 70% accuracy, 50 images → 90% accuracy (diminishing returns).

### Challenge 3: Failure Mode Root Cause Analysis

Collect data from failed real robot grasps:
- Video of gripper approaching
- Camera image at point of failure
- Joint states during grasp
- Gripper force over time

Diagnose each failure:
- Is it a perception error (wrong target), physics error (wrong force), or timing error (delayed response)?
- Use failure diagnostic from `sim_to_real_evaluation.py` to classify

Create a taxonomy of failure modes specific to your robot hardware.

---

## Further Reading

### On Sim-to-Real Transfer
- Tobin et al. "Domain Randomization..." (2017) — Foundational work
- Plappert et al. "Sim-to-Real Transfer..." (2018) — Applied to manipulation
- OpenAI Dactyl white paper — Extreme domain randomization

### On Hardware-in-the-Loop
- "Iterative Sim-to-Real Transfer" — How to structure HiL loop effectively
- Boston Dynamics blog posts on simulation infrastructure

### On Sensor Fusion
- Welch & Bishop "An Introduction to Kalman Filtering" (2006)
- Thrun "Probabilistic Robotics" (2005) — Comprehensive reference

---

## Summary

You've now learned:

✅ How to identify sim-to-real gaps in specific tasks
✅ How to design domain randomization strategies
✅ How to iterate using hardware-in-the-loop testing
✅ How to analyze real robot failures and improve sim
✅ How to evaluate transfer success quantitatively

These exercises apply the theory from Module 3 to real robotics problems. The goal is **systematic sim-to-real transfer**, not magic solutions.

---

**Next Steps:**
1. Apply these exercises to your own robot/task
2. Work through challenge exercises for deeper understanding
3. Read the papers cited in "Further Reading"
4. Proceed to Module 4 (if available) for other robotics domains

Congratulations! You now understand how to deploy simulation-trained policies to real robots reliably.

---

**Next Module**: [Module 4: VLA Systems](../vla-systems/intro.md) (Coming soon)
