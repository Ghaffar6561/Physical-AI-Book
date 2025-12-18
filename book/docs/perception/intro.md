# Module 3: Perception & Sim-to-Real Transfer

## Learning Objectives

By the end of this module, you will be able to:

**Core Concepts:**
- Explain how robots perceive their environment using multiple sensors
- Understand the perception-control loop and real-time constraints
- Identify the critical gaps between simulation and reality
- Apply domain randomization to improve sim-to-real transfer
- Design and evaluate simulation adequacy for specific tasks
- Know when to use NVIDIA Isaac for advanced applications

**Practical Skills:**
- Implement sensor fusion (combining camera, LiDAR, IMU data)
- Write domain randomization code for Gazebo
- Evaluate sim-to-real transfer success metrics
- Create fine-tuning strategies for real hardware
- Assess simulation fidelity using structured checklists
- Build hardware-in-the-loop validation pipelines

**Evaluation:**
- Identify 3+ sim-to-real gaps in a robotic application
- Design appropriate domain randomization strategies
- Create and execute sim-to-real transfer validation checklist
- Propose fine-tuning and adaptation techniques

## Module Overview

This module addresses the central challenge in robotics: **how do we take what works in simulation and make it work on real robots?**

You've built a humanoid in simulation (Module 2). You've written control code that works perfectly in Gazebo. But when you deploy to real hardware, everything breaks. The gripper is less stable. The walking gait is slower. Vision fails under different lighting. This is the **sim-to-real transfer problem**, one of the most studied challenges in robotics today.

This module teaches you to:

1. **Understand perception pipelines** — How sensors provide data to the robot
2. **Recognize simulation gaps** — What doesn't match reality
3. **Bridge the gap strategically** — Domain randomization, fine-tuning, hardware-in-the-loop
4. **Evaluate transfer success** — Metrics and checklists to validate adequacy
5. **Use advanced tools** — NVIDIA Isaac for photorealistic simulation

---

## The Perception Problem

### Why Perception Matters

A humanoid robot making decisions requires three things:

```
Sensor Data → Perception (filtering, fusion) → Robot Decision → Control
    ↑                                                            ↓
    └────────────── Feedback Loop ────────────────────────────┘
```

**Perception is the input to this loop.** If perception is wrong, decisions are wrong.

**Example**: A humanoid trying to grasp a water bottle

```
Real Robot:
  Sensor: RGB camera image
  Perception: Detect bottle in image (CNN) → 3D position (triangulation)
  Decision: "Bottle is 0.5m away, 10° to the right"
  Control: "Move arm to (x=0.5, y=0.1, z=0.9)"
  Result: ✓ Successful grasp OR ✗ Misses bottle (perception was wrong)

Simulation:
  Sensor: Perfect rendered image (no noise, perfect lighting)
  Perception: Detect bottle in image → 3D position
  Decision: "Bottle is exactly 0.5m away, 10° to the right"
  Control: "Move arm to (x=0.5, y=0.1, z=0.9)"
  Result: ✓ Perfect grasp (simulation has no friction, timing, or lighting issues)
```

The humanoid's **perception** is the weakest link in sim-to-real transfer. Why?

1. **Simulation renders perfect images** — Real cameras have noise, distortion, lag
2. **Deep learning networks trained on perfect sim data** → Fail on real noisy images
3. **Real world has infinite variations** — Lighting, occlusion, object appearance
4. **Timing is different** — Sim has zero latency; real camera has 33+ ms lag

### The Perception-Control Latency Loop

Real-time robots must close their feedback loop faster than the task dynamics.

```
Walking Robot (1 m/s):
  - Sensor update: 30-100 Hz (10-33 ms)
  - Perception: 50-200 ms (slow CNNs)
  - Decision: 10-20 ms (planning)
  - Control: 10-20 ms (motor commands)
  - Latency budget: 50-100 ms total acceptable

  Problem: If perception takes 200 ms, the robot is blind mid-step!
```

**Latency budget constraint**: Perception + planning + control < (1 / task frequency)

For a humanoid walking at 1 Hz step rate (1 step/second):
- Max total latency: 1000 ms
- But perceptual latency must fit within control loop: ~100 ms

For a robot catching a ball falling at 10 m/s:
- Max latency: 100 ms (ball falls 1 m in that time)
- Perception budget: 30-50 ms
- This is aggressive and requires specialized hardware

---

## The Sim-to-Real Gap

Even with perfect simulation, reality differs. The gaps are systematic and well-characterized:

### Gap 1: Sensor Simulation

**Simulation:**
```
Camera delivers:
  - Perfect 640×480 RGB image
  - Zero noise, exact colors
  - 0 ms latency (or configured constant lag)
  - Perfect focus and exposure
```

**Reality:**
```
Camera delivers:
  - Rolling shutter (different rows at different times)
  - Gaussian + Poisson noise in pixel values
  - 30-100 ms latency (variable)
  - Auto-exposure adapts to lighting
  - Motion blur from fast robot movement
  - Lens distortion (barrel, chromatic aberration)
```

**Impact**: A CNN trained on perfect sim images fails 50%+ of the time on real images.

**Mitigation**: Domain randomization on sensors
```python
# Sim camera: add realistic noise
camera.noise = GaussianNoise(stddev=0.01)
camera.rolling_shutter = True
camera.exposure_time = 33  # ms
```

### Gap 2: Dynamics and Physics

**Simulation:**
```
Joint control:
  - Command: joint_angle = 0.5 rad
  - Result: joint_angle becomes 0.5 rad instantly (or with configured delay)
  - Friction: μ = 0.5 (constant)
  - Contact: Perfect Coulomb model
```

**Reality:**
```
Joint control:
  - Command: joint_angle = 0.5 rad
  - Motor accelerates over 10-50 ms
  - Backlash causes overshoot/undershoot
  - Friction is velocity-dependent: μ(v) ≠ constant
  - Contact forces have complex dynamics (multiple contact points)
```

**Impact**: Aggressive control that works in sim becomes unstable on real hardware.

**Mitigation**: Conservative control + trajectory filtering
```python
# Real robot controller
max_velocity = 0.5 * sim_max_velocity  # Safety margin
max_acceleration = 0.8 * sim_max_acceleration
# Filter commands through low-pass filter
filtered_command = smooth(command, cutoff=5 Hz)
```

### Gap 3: Environmental Variation

**Simulation:**
```
Grasping a bottle:
  - Bottle always at exact position: (x=0.5, y=0.1, z=0.9)
  - Bottle material: steel with μ=0.7
  - Lighting: perfect directional light at fixed angle
  - Table texture: smooth and uniform
```

**Reality:**
```
Grasping a bottle:
  - Bottle position ± 2 cm (manufacturing variation)
  - Bottle material varies: glass (μ=0.4), plastic (μ=0.6), wet (μ=0.2)
  - Lighting: fluorescent, varies with time of day, shadows
  - Table texture: dust, scratches, spills
```

**Impact**: Gripper force tuned in sim doesn't work on different materials.

**Mitigation**: Domain randomization on objects
```python
# Randomize friction, position, appearance
friction = random.uniform(0.3, 0.8)
position = nominal_position + random.normal(0, 0.02)
color = random.choice([red, blue, green, brown])
```

### Gap 4: Timing and Synchronization

**Simulation:**
```
Perception → Decision → Control loop
  - All deterministic
  - Fixed timestep integration
  - Perfect synchronization
  - No jitter or dropped frames
```

**Reality:**
```
Perception → Decision → Control loop
  - Asynchronous (frames arrive at irregular times)
  - Variable latency (network delays, CPU load)
  - Jitter (variance in timing)
  - Dropped frames (network loss, busy CPU)
```

**Impact**: Policies trained on deterministic sim assume perfect timing. Real timing variance causes instability.

**Mitigation**: Add timing realism to sim
```python
# Simulate realistic latencies
camera_latency = 33 + random.normal(0, 5)  # ms
network_latency = 10 + random.exponential(5)  # ms
perception_latency = 100 + random.normal(0, 20)  # ms
```

---

## The Sim-to-Real Transfer Strategy

Modern robotics uses a **three-pronged approach** to bridge the gap:

### Strategy 1: Domain Randomization (in Simulation)

Train with diverse parameters so the policy is robust to variation:

```python
# Example: Randomize friction, lighting, object position
for episode in range(1000):
    # Randomize world
    friction_coeff = random.uniform(0.2, 1.0)
    light_direction = random.normal(default_direction, std=0.1)
    object_position = nominal_position + random.normal(0, 0.05)

    # Train policy
    policy.train_episode(
        friction=friction_coeff,
        light_direction=light_direction,
        object_pos=object_position
    )
```

**Result**: Policy learns to succeed despite variation. When deployed on real hardware (which is just another variant), it works.

### Strategy 2: Hardware-in-the-Loop Testing

Test on real hardware early, iterate with simulation:

```
Week 1: Train in sim with domain randomization (1000+ hours of data)
Week 2: Test on real robot (1 hour of expensive data)
    → Identify failure modes
    → Update simulator based on failures
Week 3: Retrain in improved sim with real data
Week 4: Test again on real robot
    → If successful, deploy
    → If not, repeat
```

### Strategy 3: Fine-Tuning on Real Data

Adapt the sim-trained policy using small amounts of real data:

```python
# Phase 1: Train in sim (99% of learning)
policy = train_in_simulation(domain_randomization=True)

# Phase 2: Fine-tune on real data (1% of learning)
real_data = collect_real_robot_data(hours=1)
policy = fine_tune(policy, real_data, num_epochs=10)

# Phase 3: Deploy
policy.deploy_on_real_robot()
```

---

## Simulation Adequacy: When Is Sim Enough?

Not every task needs photorealistic simulation (NVIDIA Isaac). Gazebo is often sufficient. The question is: **Is my simulation adequate for this task?**

A simulation is **adequate** if a policy trained in sim works on real hardware ≥80% of the time.

A simulation is **insufficient** if:
- Policy fails >20% on real hardware
- Failures are due to simulation mismatches (not bugs)
- Collecting real data is slow/expensive

### Adequacy Checklist

For your task, check each item:

**1. Perception:**
- [ ] Does your neural network train on sim-like data?
- [ ] Do you add realistic sensor noise to training data?
- [ ] Is lighting variation in your training set?
- [ ] Does your network handle real image artifacts (blur, distortion)?

**2. Physics:**
- [ ] Do your inertias match CAD specs ±10%?
- [ ] Did you measure real friction coefficients?
- [ ] Are contact dynamics modeled (collision response)?
- [ ] Do you account for motor saturation (velocity/torque limits)?

**3. Timing:**
- [ ] Is your control loop frequency realistic?
- [ ] Do you add latency to perception?
- [ ] Is timing jitter simulated?
- [ ] Can your policy tolerate dropped frames?

**4. Testing:**
- [ ] Did you test on varied object sizes?
- [ ] Did you test on different materials?
- [ ] Did you test with occlusion/clutter?
- [ ] Did you measure success rate on 50+ real trials?

If you checked ≥80%, your simulation is likely adequate.

---

## Module Structure

This module is organized as:

1. **[Sensor Fusion](sensor-fusion.md)** — How robots combine sensor data
   - Camera, LiDAR, IMU integration
   - Kalman filtering basics
   - SLAM for localization

2. **[Sim-to-Real Transfer](sim-to-real-transfer.md)** — Core module on bridging gaps
   - Detailed analysis of each gap
   - Domain randomization strategies
   - Fine-tuning and hardware-in-the-loop

3. **[Isaac Workflows](isaac-workflows.md)** — Advanced photorealistic simulation
   - When to use Isaac vs Gazebo
   - Synthetic data generation
   - Isaac + ROS 2 integration

4. **[Code Examples](../../../static/code-examples/)** — Practical implementations
   - Domain randomization code
   - Sim-to-real evaluation metrics

5. **[Exercises](exercises.md)** — Apply concepts
   - Design domain randomization for a task
   - Evaluate simulation adequacy

6. **[Setup Guide](setup-isaac.md)** — Environment configuration
   - NVIDIA Isaac installation
   - Docker setup for low-end hardware

---

## Key Insight: Simulation is a Tool, Not Ground Truth

The most important concept in this module:

> **Simulation is a computational tool for exploring possible behaviors. It is not ground truth. A policy's success in simulation is necessary but not sufficient for real-world success.**

This means:

- ✅ Sim allows cheap, fast training (good)
- ✅ Sim reveals algorithm failures (good)
- ❌ Sim cannot guarantee real-world success (expected)
- ❌ Perfect sim fidelity is impossible (expected)

The goal is **sufficient fidelity**, not perfect fidelity. You want simulation accurate enough that:

1. The policy learns reasonable behaviors (not broken controls)
2. The transfer gap is manageable (requires mitigation but possible)
3. Real data collection can bridge remaining gaps (not prohibitively expensive)

---

## Learning Paths

**Quick Path (3-4 hours):**
- Read: Sensor Fusion (overview only)
- Read: Sim-to-Real Transfer (focus on adequacy checklist)
- Exercise: Evaluate a given task
- Outcome: Can assess whether sim is adequate

**Standard Path (6-8 hours):**
- Read all three core sections
- Study code examples
- Exercises 1-2: Identify gaps, design randomization
- Outcome: Can design sim-to-real strategies for simple tasks

**Deep Dive Path (12+ hours):**
- Read all sections carefully
- Implement domain randomization code
- Trace through hardware-in-the-loop workflows
- Exercises 1-3: Full simulation→real transfer cycle
- Challenge exercises: Advanced topics
- Outcome: Can architect sim-to-real pipelines from scratch

---

## Glossary

**Perception Pipeline** — Chain of sensor → filtering → feature extraction → decision

**Sensor Fusion** — Combining data from multiple sensors to reduce uncertainty

**Domain Randomization** — Training with varied parameters to build robustness

**Sim-to-Real Transfer** — Deploying a sim-trained policy on real hardware

**Simulation Gap** — Difference between sim physics/sensors and reality

**Simulation Fidelity** — How accurately simulation matches reality

**Fine-Tuning** — Adapting a sim-trained model using real data

**Hardware-in-the-Loop** — Testing with real hardware in the control loop

**Latency Budget** — Maximum acceptable delay in perception-control loop

**Domain Shift** — Change in data distribution (sim → real)

**Adequate Simulation** — Sim where transfer success ≥80% without additional data collection

---

## Assumptions & Prerequisites

**You should know:**
- ROS 2 nodes and topics (Module 1)
- URDF robot modeling (Module 2)
- Gazebo physics simulation (Module 2)
- Basic control theory (feedback loops, PID)
- Python programming
- Neural networks (CNNs for perception)

**You'll learn:**
- Sensor fusion and filtering
- Domain randomization techniques
- Sim-to-real transfer strategies
- NVIDIA Isaac workflows
- Evaluation metrics for transfer success

**Tools you'll use:**
- Gazebo (already installed)
- Python + NumPy/SciPy
- PyTorch or TensorFlow (optional, for NN-based perception)
- NVIDIA Isaac (for advanced sections)
- ROS 2 CLI tools

---

## The Real Challenge

By the end of this module, you'll understand that **sim-to-real transfer is not magic**. It requires:

1. **Careful simulation design** — Capture the important physics
2. **Domain randomization** — Build robustness systematically
3. **Real hardware testing** — Identify and fix the unexpected gaps
4. **Iterative refinement** — Update sim based on real-world failures

This is why leading robotics labs (Boston Dynamics, OpenAI, DeepMind) spend as much time on **simulation infrastructure** as on algorithm development. The simulation isn't just a training tool—it's the foundation for reproducible, scalable robotics.

---

**Next**: [Sensor Fusion](sensor-fusion.md) — Learn how robots combine multiple sensors to perceive their world.

You now understand why this module matters. Simulation alone isn't enough. Real robots live in the real world, and we need strategies to bridge that gap systematically.
