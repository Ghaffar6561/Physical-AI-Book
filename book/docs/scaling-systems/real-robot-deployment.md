# Real Robot Deployment: From Simulation to Production

The hardest problem in robotics: **making simulation work on real robots.**

---

## The Sim-to-Real Gap

### What Works Perfectly in Simulation

```
Gazebo Physics:
  ✓ Perfect sensor data (no noise)
  ✓ Precise actuators (follow commands exactly)
  ✓ No wear/tear
  ✓ Deterministic execution
  ✓ Instant reset between episodes

Result: Policy trained in sim achieves 95% success
```

### What Breaks on Real Robots

```
Real Robot Problems:
  ✗ Camera noise (rolling shutter, exposure changes)
  ✗ Sensor drift (joint encoders accumulate error)
  ✗ Actuator lag (gripper takes 200ms to respond)
  ✗ Friction variability (table surface changes)
  ✗ Object variation (similar cup: different weight/friction)
  ✗ Timing issues (network latency, compute variability)
  ✗ Wear/tear (gripper loosens over time)

Result: Policy trained in sim achieves 30% success on real robot
```

### The Catastrophic Failure Case

```
Simulation: Grasp → Success (95%)

Real Robot: Grasp → Failure (30%)

Why? Policy learned features specific to sim:
  - Knows walls are at specific positions
  - Assumes exact object colors
  - Expects zero sensor noise
  - Never saw worn gripper

On real robot: Different lighting, wall positions, worn parts
              All features suddenly invalid → failures
```

---

## Domain Randomization: The Fix

### Core Idea

```
"If robot trained on thousand different environments,
 will generalize to real world (one more environment)"
```

### What to Randomize

```
CRITICAL (must randomize):
  ├─ Object colors: Random RGB per episode
  ├─ Object size/shape: Vary dimensions ±10%
  ├─ Object friction: μ ∈ [0.3, 1.5]
  ├─ Table height: ±5cm
  ├─ Lighting: Brightness, shadows, colors
  └─ Camera position: Small variations

IMPORTANT (randomize if possible):
  ├─ Object weight: ±20% of nominal
  ├─ Gripper friction: Varies with wear
  ├─ Joint stiffness: Motors age differently
  └─ Delay: Network latency 0-100ms

OPTIONAL (randomize if time):
  ├─ Background clutter
  ├─ Texture variations
  └─ Camera distortion (lens aberrations)
```

### Implementation

```python
def randomize_environment():
    """Randomize simulation parameters each episode."""

    # Visual randomization
    object_color = random_color()  # (R, G, B)
    object_size = random(0.9, 1.1) * nominal_size
    table_color = random_color()

    # Physics randomization
    friction = random(0.3, 1.5)
    gravity = random(9.6, 9.9)
    mass = random(0.8, 1.2) * nominal_mass

    # Sensor randomization
    camera_noise = random(0, 0.02)  # Gaussian noise std
    joint_noise = random(0, 0.05)   # Encoder noise

    # Actuator randomization
    gripper_lag = random(50, 250)   # milliseconds
    torque_limit = random(0.8, 1.0) * nominal

    # Apply all to environment
    env.set_color(object_color)
    env.set_friction(friction)
    env.set_gravity(gravity)
    # ... etc for all parameters

    return env

# Training loop
for episode in range(100000):
    env = randomize_environment()  # NEW! Different each time
    trajectory = collect_episode(env, policy)
    train_policy(trajectory)
```

### Results: Domain Randomization Impact

```
Real Robot Grasping Success Rate:

No randomization:
  Simulation: 95% ✓
  Real robot: 30% ✗
  Transfer ratio: 0.31 (terrible)

With light randomization:
  Simulation: 80% (performance drop during training)
  Real robot: 55%
  Transfer ratio: 0.69 (better)

With heavy randomization:
  Simulation: 65%
  Real robot: 76% ✓
  Transfer ratio: 1.17 (exceeds sim performance!)

Takeaway:
  Train on 1000 randomized simulations = generalize to 1 real world
```

---

## Fine-Tuning on Real Data

### Strategy: Warm-Start + Fine-Tune

```
Step 1: Train thoroughly in simulation with domain randomization
  Episodes: 100K
  Time: 2-3 weeks GPU
  Result: 65% success in sim

Step 2: Collect real robot data (with caution!)
  Episodes: 100-200 (expensive!)
  Time: 2-4 hours human time
  Collect: 50% successful, 50% failed grasps

Step 3: Fine-tune on real data
  Freeze encoder: Keep learned features
  Train head: Adapt to real robot
  Data: 200 real episodes + 2000 synthetic episodes (50% real)
  Time: 1-2 days GPU
  Result: 82% success on real robot
```

### Real Data Collection Strategy

```
Phase 1: Collect Data Safely (4 hours)
  ├─ Manual grasps (robot under human control)
  ├─ Collect 200 videos of human moving gripper
  ├─ Label success/failure
  └─ Result: 200 labeled demonstrations

Phase 2: Warm-Start Collect (4 hours)
  ├─ Use simulation-trained policy
  ├─ Let robot try grasping
  ├─ Collect ALL trials (success + failure)
  ├─ Human intervention if danger
  └─ Result: 400 robot-collected demonstrations
           (many failures, natural data distribution)

Phase 3: Fine-Tune (2 days)
  ├─ Mix Phase 1 + Phase 2 data
  ├─ Train on 600 real examples
  ├─ Monitor success rate (should increase)
  └─ Result: Production-ready policy
```

---

## Production Safety: Preventing Failures

### Fallback Policies

```python
def execute_grasp_safely(policy, image, gripper_state):
    """Execute grasp with fallback safety."""

    # Step 1: Get policy prediction
    action = policy.predict(image)

    # Step 2: Check feasibility
    if not is_safe(action):
        print("Unsafe action, using fallback")
        return fallback_action()

    # Step 3: Execute with monitoring
    try:
        execute_action(action, timeout=5.0)
    except (Timeout, Collision, GripperStall):
        print("Execution failed, recovery mode")
        return recover_gripper()

    # Step 4: Verify success
    if not verify_success(gripper_state):
        return retry_with_adjusted_grip()

    return True
```

### Constraint Checking

```python
def is_safe(action):
    """Check action violates no constraints."""

    x, y, z, gripper_width = action

    # Joint limits
    if x < -0.5 or x > 0.5:  # Reachability
        return False

    # Collision avoidance
    if z < 0.05:  # Would hit table
        return False

    # Gripper limits
    if gripper_width < 0.02 or gripper_width > 0.10:
        return False

    # Speed limits (no jerky motions)
    if abs(action - previous_action).max() > 0.2:
        return False

    return True
```

### Graceful Degradation

```
Normal Operation:
  Policy prediction: Use full learned policy

Degraded Mode 1 (Vision Fails):
  Camera error → Use last known good image
  Fall back to: Move to home position

Degraded Mode 2 (Gripper Fails):
  Gripper stuck → Stop immediately
  Fall back to: Release object, raise arm

Degraded Mode 3 (Network Slow):
  Inference taking >1 second
  Fall back to: Use cached prediction from 2 steps ago

Degraded Mode 4 (Many Failures):
  >3 failures in a row → Call human
  Disable autonomous operation until fixed
```

---

## Hardware Integration: ROS 2 Integration

### Real-Time Requirements

```
Real-time task: Gripper control at 100 Hz
  Max latency: 10ms

Latency breakdown:
  Perception (5ms):  Image capture + preprocessing
  Inference (200ms):  Network forward pass ✗ PROBLEM!
  Gripper command (5ms): Send to robot

Solution: Reduce inference latency
  ├─ Smaller network (2M instead of 50M params)
  ├─ Quantization (8-bit instead of 32-bit)
  ├─ Batch inference (process 5 frames at once)
  └─ Result: 50ms inference → acceptable

Alternative: Predictive control
  ├─ Predict 5 steps ahead
  ├─ Execute predictions while computing next
  ├─ Inference latency becomes invisible
  └─ Requires smooth trajectories (use diffusion policy)
```

### Monitoring & Logging

```python
class RobotController:
    def __init__(self, policy):
        self.policy = policy
        self.logger = setup_logging()

    def control_loop(self, rate=100):  # 100 Hz
        """Main robot control loop."""
        while True:
            # Perception
            image = camera.capture()
            proprioception = robot.get_state()

            # Inference
            start = time.time()
            action = self.policy.predict(image)
            latency = time.time() - start

            # Log metrics
            self.logger.log({
                'timestamp': time.time(),
                'action': action,
                'inference_latency_ms': latency * 1000,
                'gripper_position': proprioception['gripper'],
            })

            # Execute
            robot.execute_action(action, timeout=1.0)

            # Sleep to maintain rate
            sleep_time = 1.0/100 - (time.time() - start)
            time.sleep(max(0, sleep_time))

    def analyze_logs(self):
        """Analyze logged data for failures."""
        df = self.logger.load_logs()

        # Latency analysis
        latencies = df['inference_latency_ms']
        print(f"Mean latency: {latencies.mean():.1f}ms")
        print(f"95th percentile: {latencies.quantile(0.95):.1f}ms")

        # Failure analysis
        failures = df[df['success'] == False]
        print(f"Failure rate: {len(failures)/len(df)*100:.1f}%")
```

---

## Sim-to-Real Checklist

Before deploying to real robot:

- [ ] Domain randomization implemented (colors, physics, sensors)
- [ ] Simulation success rate > 70%
- [ ] Real robot test run (5 trials) shows >30% success
- [ ] Constraint checking code written
- [ ] Fallback policy implemented
- [ ] Recovery procedures defined (gripper stuck, arm collides)
- [ ] Logging system set up
- [ ] Latency < 500ms (or use predictive control)
- [ ] Safety reviewed by second person
- [ ] Insurance check (robot can cause harm?)

---

## Real-World Case Study: Boston Dynamics Spot

### Deployment Scenario

```
Warehouse:
  ├─ Unstructured environment (boxes, obstacles)
  ├─ Varying lighting (skylights, shadows)
  ├─ Temperature fluctuations (50-80°F)
  ├─ Different operators (inconsistent positioning)
  └─ Long deployment (weeks without intervention)
```

### Solutions Implemented

```
1. Heavy Domain Randomization
   ├─ Random lighting (0-200% brightness)
   ├─ Random obstacles (boxes, walls)
   ├─ Random surfaces (tile, carpet, concrete)
   └─ Result: Works across warehouse locations

2. Adaptive Control
   ├─ Monitor joint currents (detect obstacles)
   ├─ Predict and avoid collisions
   ├─ Graceful degradation (navigate around)

3. Continuous Monitoring
   ├─ Log all operations
   ├─ Detect failures (unexpected joint angles)
   ├─ Request human help when confused

4. Hardware Robustness
   ├─ Redundant sensors (multiple cameras)
   ├─ Stiff actuators (won't break easily)
   ├─ Regular maintenance (grease joints, check alignment)
```

### Results

```
Spot Deployment Metrics:
  ├─ Uptime: 99.2% (down 7 hours/month for maintenance)
  ├─ Task success: 94% (self-healing, retries, fallbacks)
  ├─ Human interventions: 0.5/day (very rare)
  ├─ Cost: Self-pays via task revenue
  └─ Deployment duration: 18+ months continuous
```

---

## Key Takeaways

✅ **Domain Randomization**: Train on 1000 variations → generalize to reality
✅ **Transfer Ratio**: Real success should be 70%+ of sim success
✅ **Fine-Tuning**: 200 real examples + sim pre-training beats scratch
✅ **Safety First**: Fallbacks, constraints, monitoring prevent failures
✅ **Graceful Degradation**: Degrade gracefully when things break
✅ **Continuous Monitoring**: Logging catches failures before they cascade

---

## Next Steps

1. **Implement domain randomization** - Randomize all sim parameters
2. **Measure transfer ratio** - Real vs sim success
3. **Design fallback policies** - What to do when vision fails?
4. **Set up logging** - Monitor every aspect
5. **Deploy carefully** - Start with supervised mode, then autonomous

---

## Further Reading

- **Domain Randomization for Transferring Deep Neural Networks** (Tobin et al., 2017)
- **Closing the Sim-to-Real Loop** (Rusu et al., 2016)
- **Dactyl Hand Paper** (OpenAI, 2018): Massive domain randomization
- **Boston Dynamics Spot Real-World Reports**: https://blog.bostondynamics.com

---

**Next Section:** [Distributed Training →](distributed-training.md)

