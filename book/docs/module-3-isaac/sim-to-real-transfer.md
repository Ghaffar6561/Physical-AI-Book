# Sim-to-Real Transfer: Bridging Simulation and Reality

This is the core chapter of the Physical AI book. It explains why well-trained policies fail catastrophically on real robots and the systematic methods to fix them.

## The Core Problem: Why Simulation-Trained Policies Fail

### A Real Story

In 2016, OpenAI trained a deep reinforcement learning policy to perform robotic grasping in simulation. The policy achieved 99% success in sim. When deployed to a real UR5 robot, it failed 90% of the time. The robot's gripper would:

- Approach the object correctly (perception worked)
- Grasp at the wrong force (physics mismatch)
- Drop the object or crush it (control mismatch)
- Never adapt when the object slipped (no feedback learning)

**Why?** The simulation was "perfect" by conventional standards. Physics were accurate. Rendering looked good. But it missed hundreds of small details that don't matter individually but combine catastrophically.

This chapter is about solving this systematically.

---

## The Transfer Gap: What Changes Between Sim and Real

Revisiting the four gaps from the intro (now in detail):

### Gap 1: Sensor Fidelity

**What simulation assumes:**
```
Camera(t) = RenderImage(world_state)  // Perfect image instantly

Output:
  - RGB image: 640×480, perfect colors
  - Latency: 0 ms (or constant configured lag)
  - Noise: None (or Gaussian with fixed σ)
  - Properties: Perfect focus, exposure, motion blur = 0
```

**What real cameras do:**
```
Camera(t) = SensorNoise(RollingShutter(MotionBlur(Exposure(RenderImage()))))

Rolling shutter: Different rows scanned at different times
  Row 0:   captured at t = 0 ms
  Row 239: captured at t = 33 ms (whole frame takes 33 ms!)
  Result: Fast-moving objects appear slanted

Motion blur: Moving objects smeared across image
  Gripper moving at 0.5 m/s during frame exposure (16 ms)
  Smear distance: 0.5 m/s × 16 ms = 8 mm on image
  CNN trained on sharp images has never seen blurred gripper

Exposure control: Auto-exposure adapts to lighting
  Bright scene: Exposure ↓ (darker image, more noise)
  Dark scene: Exposure ↑ (brighter image, but saturated highlights)
  Result: Same object looks different in different lighting

Lens distortion: Barrel/pincushion distortion
  Straight lines appear curved near image edges
  CNN sees distorted training images never saw in sim
```

**Impact on grasping:**
```
Sim-trained gripper detector trained on:
  - Sharp, noise-free images
  - Consistent exposure
  - No distortion

Real gripper in different lighting:
  - Rolling shutter makes gripper fingers appear slanted
  - Motion blur if arm moving
  - Auto-exposure changes image brightness

Result: CNN confidence ↓ from 0.95 (sim) to 0.60 (real)
        → Wrong grasp points
        → Object drops
```

### Gap 2: Physics Simulation

**What simulation assumes (ODE physics engine):**
```
Friction model: f = μ * N  // Coulomb friction
  - μ constant regardless of velocity
  - Contact forces deterministic
  - No hysteresis

Motor model:
  Joint angle(t) = desired_angle  // Instant response

  In reality, motor acceleration-limited:
    τ = I * dω/dt
    Max torque: 100 Nm
    Motor inertia: 0.5 kg⋅m²

    Time to reach 1 rad/s: τ / I = 100 / 0.5 = 200 rad/s²
    So reaching 1 rad/s takes: 1 rad/s / 200 rad/s² = 5 ms

  In sim: INSTANT ✗
```

**What real physics does:**
```
Friction model: f(v) = μ_s * N (when v=0, static)
                       μ_k * N * tanh(v/v_ref) (kinetic)

  Example: Grasping wet bottle
    Sim friction: μ = 0.7 (constant)
    Real friction: μ = 0.3 (wet, slippery)
    Required grasp force (sim): F = 50 N
    Actual slip force (real): F_slip = 30 N

    Result: Gripper set for 50 N, bottle slips at 30 N
            → Drop ✗

Motor model: Real motor has limits
  Max torque: 100 Nm (saturation)
  Max velocity: 10 rad/s
  Max acceleration: 200 rad/s²

  Sim ignores these → controller goes unstable on real hardware
```

### Gap 3: Environmental Variation

**What simulation assumes:**
```
Object properties: Fixed parameters
  Bottle position: exactly (0.5, 0.1, 0.9) meters
  Bottle mass: exactly 1.0 kg
  Bottle friction: exactly μ = 0.7
  Table friction: exactly μ = 0.8

Lighting: Fixed, predictable
  Sun direction: always (1, 0, 1) normalized
  Sun intensity: constant
  Ambient light: constant
  Shadows: predictable

Appearance: Consistent
  Bottle color: always red RGB(255, 0, 0)
  Texture: identical rendering every time
  Material: shiny plastic (fixed)
```

**What reality does:**
```
Object variation:
  Position: (0.5 ± 0.02, 0.1 ± 0.02, 0.9 ± 0.01) m
            (manufacturing tolerance ±2 cm)

  Mass: 1.0 ± 0.05 kg (3% variation in bottle fill level)

  Friction: Different bottles, different materials
    Glass bottle: μ = 0.4 (slippery)
    Plastic: μ = 0.6 (grippier)
    Wet: μ = 0.2 (very slippery)

  Table: Dust, scratches, spills
    Friction: μ = 0.8 to μ = 0.3 (with dust/water)

Lighting: Constantly changing
  Time of day: Morning (blue sunlight) vs evening (orange)
  Weather: Sunny vs cloudy (intensity changes)
  Indoor: Fluorescent (strong shadows) vs LED (diffuse)
  Shadows: Cast by robot, other objects, windows

Appearance: Natural variation
  Bottle: Dust, fingerprints, labels partially visible
  Lighting: Reflections change with viewing angle
  Occlusion: Other objects partially hide target
```

**Example: Gripper force failure**
```
Scenario: Grasp a bottle

Sim training:
  Bottle always: 1.0 kg, μ=0.7, clean
  Optimal grasp force: 50 N (learned by policy)
  Success rate: 98% in sim ✓

Real deployment:
  Bottle 1: 1.0 kg, μ=0.6, clean       → 50 N works ✓
  Bottle 2: 0.95 kg, μ=0.5, dusty      → 50 N works ✓
  Bottle 3: 0.8 kg, μ=0.2, wet         → 50 N FAILS (slip) ✗
  Bottle 4: Different bottle shape       → Gripper geometry mismatch ✗

Real success: 60% (not 98%)
```

### Gap 4: Timing and Synchronization

**What simulation assumes:**
```
Control loop frequency: Deterministic 50 Hz
  All components sync perfectly
  Camera arrives at t=0ms, t=20ms, t=40ms, ... (exact)
  Perception runs for exactly 10 ms
  Decision runs for exactly 5 ms
  Control runs for exactly 5 ms
  Total latency: exactly 20 ms ✓

Dropped frames: Never
Frame jitter: Zero
```

**What reality does:**
```
Control loop frequency: Asynchronous
  Camera frame 1: arrives at t=0 ms
  Camera frame 2: arrives at t=35 ms (not 20!)

  CPU load varies:
    Desktop: 20% → perception takes 10 ms
    Background process starts → perception takes 50 ms
    Result: Latency varies 10-50 ms (5× variation!)

Dropped frames: Occasional
  Network loses packet → frame arrives corrupted
  Gripper is busy → delayed response
  Result: Missing one frame throws off timing estimates

Jitter: High variance in timing
  Frame arrival: {20, 22, 18, 25, 19, 23} ms spacing
  (Not always exactly 20 ms)

  Effect: Kalman filter covariance increases
          Robot loses confidence in position estimate
          Navigation becomes unreliable
```

---

## Strategy 1: Domain Randomization

The most successful method for sim-to-real transfer today. Train the policy with randomized parameters so it becomes robust to variation.

### The Core Idea

```
Standard training (deterministic):
  Every episode: Same world, same physics
  Result: Policy overfits to exact simulation
          Cannot handle any variation

Domain randomization training:
  Every episode: Random world parameters
    Random friction: μ ∈ [0.2, 1.0]
    Random object size: radius ∈ [0.03, 0.08] m
    Random lighting: direction ∈ random sphere
    Random camera noise: σ ∈ [0.0, 0.05]
    Random motor delays: delay ∈ [0, 50] ms
    ...

  Result: Policy learns to succeed despite variation
          When deployed on real hardware (just another variation), it works!
```

### Example: Domain Randomization for Grasping

```python
def create_randomized_world():
    """Generate a randomized Gazebo world for training."""

    # Friction randomization
    friction_coeff = np.random.uniform(0.2, 1.0)

    # Object size randomization
    obj_radius = np.random.uniform(0.03, 0.08)

    # Object position randomization
    obj_x = 0.5 + np.random.normal(0, 0.05)
    obj_y = 0.1 + np.random.normal(0, 0.05)

    # Lighting randomization
    light_x = np.random.uniform(-1, 1)
    light_y = np.random.uniform(-1, 1)
    light_z = np.random.uniform(0.5, 2.0)
    light_intensity = np.random.uniform(0.5, 1.5)

    # Camera noise randomization
    camera_noise_std = np.random.uniform(0.0, 0.05)

    # Motor latency randomization
    motor_delay_ms = np.random.uniform(0, 50)

    # Gripper friction randomization
    gripper_friction = np.random.uniform(0.3, 0.9)

    # Return randomized parameters
    return {
        'friction': friction_coeff,
        'object_radius': obj_radius,
        'object_position': (obj_x, obj_y, 0.9),
        'light_direction': (light_x, light_y, light_z),
        'light_intensity': light_intensity,
        'camera_noise': camera_noise_std,
        'motor_delay_ms': motor_delay_ms,
        'gripper_friction': gripper_friction,
    }


# Training loop
for episode in range(1000):
    # Generate random world
    params = create_randomized_world()

    # Update Gazebo world with random parameters
    update_gazebo_world(params)

    # Train policy on this randomized episode
    policy.train_episode(world_params=params)
```

### What to Randomize?

**Critical randomizations (must include):**
- **Object properties**: Mass (±10%), friction (±30%), size (±10%)
- **Lighting**: Direction, intensity, color temperature
- **Camera noise**: Gaussian noise (σ = 0-5%), rolling shutter, motion blur
- **Motor delays**: Uniform 0-50 ms
- **Sensor latencies**: Camera (30-50 ms), LiDAR (0-100 ms)

**Optional randomizations (helps robustness):**
- **Object appearance**: Color, texture, reflectivity
- **Table friction**: Different surfaces
- **Gravity**: Small variations (±2%)
- **Wind**: For flying robots
- **Contact dynamics**: Different collision models

**Not recommended (breaks learning):**
- **Task fundamentals**: Don't randomize the goal
- **Robot morphology**: Don't change arm length randomly
- **Physics engine**: Use same engine for training and testing

### How Much Randomization?

**Rule of thumb**: Randomization ranges should roughly match real-world variation

```
Parameter               Sim Default    Randomization Range    Real-World Range
─────────────────────────────────────────────────────────────────────────────
Object friction         μ = 0.7        [0.3, 1.0]            [0.2, 1.0]
Motor delay             0 ms           [0, 50] ms            [5, 50] ms
Camera noise            0              σ ∈ [0, 0.05]         σ ∈ [0.02, 0.1]
Object mass variation   0%             ±20%                  ±10-30%
Lighting intensity      1.0            [0.5, 1.5]            [0.3, 2.0]
```

### Domain Randomization Success: Real Results

**OpenAI Robotic Hand (Dactyl) 2018:**
- Trained: 100% in sim with extreme randomization
- Real robot: 76% success on complex hand manipulation
- Key: Randomized friction, object properties, lighting, camera noise
- One of the first major sim-to-real successes

**Sim2Real Transfer for Robot Grasping (Mahler et al., UC Berkeley):**
- Trained: Grasping CNN in simulation
- Real robot (ABB arm + Suction gripper): 70% success
- 4×–5× improvement over non-randomized baseline
- Key: Randomized photorealistic rendering, friction

---

## Strategy 2: Hardware-in-the-Loop (HiL) Testing

Train in simulation, test on real hardware early and often. Use real failures to improve the simulation.

### The Workflow

```
Week 1: Train in Simulation
  - Start with baseline domain randomization
  - Run 1000 episodes (could take 1-10 hours depending on complexity)
  - Achieve 95% success in sim

Week 2: First Hardware Test
  - Deploy policy to real robot
  - Run 100 trials (expensive: 1 hour of real robot time)
  - Measure success rate: 60% (not 95!)

Analyze failures:
  - Video failures: Gripper too slow, overshoots, can't reach
  - Sensor failures: Object detection fails in real lighting
  - Physics failures: Grasps with wrong force
  - Timing failures: Commands arrive out of order

Week 3: Improve Simulation
  - Add motor delay: was [0, 50ms], now [10, 100ms] (more realistic)
  - Add camera noise: was σ=0.05, now σ=0.1 (more realistic)
  - Measure real motor response curves, add to sim
  - Randomize gripper friction more aggressively

Week 4: Retrain in Improved Simulation
  - Train again with updated randomization
  - Now achieve 90% in sim (lower because harder)
  - Deploy to real robot: 85% success!

Week 5-6: Iterate
  - More hardware testing
  - More sim improvements
  - Converge to 90%+ real success
```

### Why This Works

1. **Simulation gaps are discoverable** — Real failures reveal what's missing
2. **Simulation improvement is cheap** — Adding parameters costs nothing
3. **Training is fast** — Can retrain overnight
4. **Convergence is guaranteed** — Each iteration brings reality closer

### Implementation: Hardware-in-Loop Infrastructure

Real companies invest heavily in HiL infrastructure:

**Boston Dynamics Spot:**
- Large robot farm with 10-20 identical robots
- Parallel training: Train policy on sim fleet (10 simulations in parallel)
- Daily HiL validation: Test best policy on real robot daily
- Feedback loop: Real failures → sim improvements → next day's training

**Tesla Humanoid Project:**
- Fleet of humanoids at Tesla factories
- Real grasping tasks (part assembly)
- Sim-trained policy → deployed → real failures logged
- Failures → synthetic data generation → retrain
- Cycle time: 1-2 weeks per iteration

**Typical Setup (for researchers):**
```
┌─────────────────────────────────────────────────────────┐
│ Lab Infrastructure                                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Simulation Farm              Real Hardware              │
│ ┌───────────────────────┐  ┌──────────────────────┐    │
│ │ 4 GPU instances       │  │ 1 real robot         │    │
│ │ Running sim training  │  │ Test policy (hourly) │    │
│ │ (4 parallel envs)     │  │ Log failures         │    │
│ └───────────────────────┘  └──────────────────────┘    │
│          ↓                          ↓                    │
│  Policy checkpoint/hour    Success metrics              │
│          ↓                          ↓                    │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Analysis Loop (overnight)                          │  │
│ │ - Identify failure modes from real robot video    │  │
│ │ - Update sim parameters (friction, noise, etc.)   │  │
│ │ - Retrain overnight                                │  │
│ └────────────────────────────────────────────────────┘  │
│          ↓                                               │
│ Next morning: New policy ready for testing              │
└─────────────────────────────────────────────────────────┘
```

---

## Strategy 3: Fine-Tuning on Real Data

After sim pre-training, adapt the policy using small amounts of real robot data.

### The Approach

```
Phase 1: Pre-train in simulation (99% of learning)
  - 1000 episodes in sim
  - Achieved policy: 95% accuracy
  - Cost: ~10 GPU-hours

Phase 2: Collect real data (expensive)
  - Run policy on real robot
  - Collect 100 successful trials (30 min of robot time)
  - Videos + sensor data logged
  - Cost: ~$100-1000 (robot-specific)

Phase 3: Fine-tune on real data (1% of learning)
  - Use real data to adapt perception/control
  - Not full training (would need 1000 real examples)
  - Just minor adjustments to sim-trained policy
  - Number of epochs: 5-20 (not 500)
  - Cost: ~1 GPU-hour

Result: Sim-trained policy → fine-tuned → 90% real success
```

### Fine-Tuning Strategy by Component

**For perception (vision models):**
```python
# Pre-train in sim on 10,000 synthetic images
model = train_cnn_perception(sim_images=10000)  # 95% accuracy in sim

# Collect real data: 100 real images from actual object grasps
real_images, real_labels = collect_real_robot_data(100)

# Fine-tune: Adjust last 2 layers on real data
for epoch in range(10):
    loss = model.fine_tune(real_images, real_labels)

# Result: Model now 88% accurate on real objects (up from 50%)
```

**For control (policy networks):**
```python
# Pre-train in sim
policy = train_policy_sim(1000)  # 95% success rate in sim

# Collect real data: 50 real grasps with recorded sensor feedback
real_trajs = collect_real_robot_trajectories(50)

# Fine-tune: Adjust state-action value function
for epoch in range(5):
    policy.fine_tune(real_trajs, learning_rate=0.0001)  # Small LR

# Result: Policy now 85% successful on real robot
```

### When Fine-Tuning Works Best

✅ **Works well:**
- Perception model trained on synthetic data, fine-tuned on real images
- Robot already trained in sim; minor real-world adaptation
- Task variation small (e.g., different gripper types on same robot)

❌ **Doesn't work:**
- Fundamental sim-to-real gap too large
- Real task completely different from sim (e.g., bipedal walking → different physics)
- Not enough real data to fine-tune (law of diminishing returns)

---

## Putting It Together: Complete Sim-to-Real Pipeline

### Reference Implementation: From Simulation to Deployment

```
Step 1: Design robust simulation
  ✓ Include all important physics
  ✓ Create domain randomization
  ✓ Add realistic sensor simulation

Step 2: Train in simulation with randomization
  ✓ 1000+ episodes
  ✓ Achieve 95%+ success rate
  ✓ Save checkpoint

Step 3: Hardware-in-the-loop test
  ✓ Deploy best checkpoint to real robot
  ✓ Run 100 trials
  ✓ Log failures with video

Step 4: Analyze and improve
  ✓ Watch failure videos
  ✓ Identify missing sim parameters
  ✓ Update randomization ranges
  ✓ Retrain

Step 5: Fine-tune on real data (optional)
  ✓ Collect 100+ real successful trajectories
  ✓ Fine-tune perception/control
  ✓ Retest on real robot

Step 6: Deploy
  ✓ 90%+ success on real robot
  ✓ Acceptable for deployment
  ✓ Monitor and log failures for next iteration
```

### Typical Timeline

```
Task: Train humanoid to grasp objects

Timeline:
  Day 1: Baseline sim training + domain randomization design
  Day 2: First hardware test (20% success)
  Day 3: Analyze failures, improve sim
  Day 4: Retrain (achieve 60% on real)
  Day 5-7: Hardware testing + iteration cycle
  Day 8: Fine-tune on real data (80% success)
  Day 9: Final validation (90% success)
  Day 10: Deploy

Total effort: 1-2 weeks for research team
```

---

## Key Metrics: Measuring Transfer Success

### Success Criteria

✅ **Transfer successful** if:
```
Real-world success rate / Sim success rate ≥ 0.80
OR
Real-world success rate ≥ 90%
```

❌ **Transfer failed** if:
```
Real-world success rate < 50%
(indicates fundamental gap in simulation)
```

### Diagnostic Metrics

**Perception-based failures:**
```
If: Real success drops from 95% → 60%
Check: Object detection accuracy in real images
  - Test CNN on real camera images
  - If accuracy < 80%, need more vision training

Solution:
  - Collect more real training images
  - Fine-tune perception model
  - Add more sensor noise to sim
```

**Physics-based failures:**
```
If: Grasps succeed in picking up but fail in holding
Check: Gripper force calibration
  - Measure real gripper force response
  - Measure object friction

Solution:
  - Add motor saturation to sim
  - Randomize friction more aggressively
  - Use force feedback control in real deployment
```

**Timing-based failures:**
```
If: Grasp position correct but grasping too slow/fast
Check: Motor latency and command buffering
  - Measure actual motor response time
  - Check message queue delays

Solution:
  - Add motor model to sim
  - Simulate realistic latency
  - Retrain with delays enabled
```

---

## Common Pitfalls and How to Avoid Them

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **Insufficient randomization** | Works in sim (98%) but fails on real (30%) | Increase randomization ranges; match real-world variation |
| **Randomization too extreme** | Sim success drops (60%), policy doesn't learn | Reduce randomization; ensure task still learnable |
| **Wrong sensor simulation** | Real camera sees different lighting, CNN fails | Add realistic camera noise, rolling shutter, exposure control |
| **Motor model mismatch** | Real gripper can't reach targets set in sim | Add motor dynamics, acceleration limits, saturation |
| **No hardware testing early** | Discover gap after 2 weeks; wasted time | Test on real robot by Day 2 |
| **Task too hard in sim** | Policy never solves sim task; can't transfer | Simplify task or add curriculum learning |
| **Only testing successful cases** | Deployment fails on edge cases | Test diverse scenarios (different object sizes, materials, positions) |

---

## Advanced Topics: Beyond Basic Randomization

### Curriculum Learning

Train with easy tasks first, then progressively harder ones:

```
Curriculum:
  Stage 1: Grasp objects 5 cm from gripper (easy)
    Success: 99%

  Stage 2: Grasp objects 15 cm away (medium)
    Success: 95%

  Stage 3: Grasp occluded objects (hard)
    Success: 85%

  Stage 4: Grasp with poor lighting (very hard)
    Success: 80%

Result: Better convergence than training all difficulties at once
```

### Adaptive Randomization

Let the learning algorithm determine randomization ranges automatically:

```
Start with large randomization: μ ∈ [0.1, 2.0]
  Success rate: 10% (too hard)

Auto-reduce difficulty:
  If success < 20%, reduce randomization

New range: μ ∈ [0.3, 1.2]
  Success rate: 60%

Gradually increase:
  Once stable at 90%, expand range again: μ ∈ [0.2, 1.5]

Result: Algorithm finds optimal randomization automatically
```

### System Identification

Measure real robot properties and inject them into sim:

```python
# Measure real gripper force response
real_gripper_force_response = measure_gripper_response()

# Fit polynomial model
force_model = fit_polynomial(real_gripper_force_response)

# Inject into sim
sim_gripper.force_model = force_model

# Retrain with accurate gripper model
policy = train_in_sim()

# Real deployment with same model: higher success
```

---

## Key Takeaways

| Strategy | When to Use | Effort | Effectiveness |
|----------|-------------|--------|----------------|
| **Domain Randomization** | Always (foundational) | Medium | 70-80% success |
| **Hardware-in-the-Loop** | When you have real robot | High | 80-95% success |
| **Fine-Tuning on Real Data** | When perception/control need adjustment | Medium | +10-15% improvement |
| **System Identification** | When robot properties known | Medium | +5-10% improvement |
| **Curriculum Learning** | For complex tasks | Medium | Better convergence |

---

## The Bottom Line

Sim-to-real transfer is not magic. It requires:

1. **Thoughtful simulation** — Include the important physics and sensors
2. **Systematic randomization** — Build robustness to real-world variation
3. **Hardware testing** — Find gaps, iterate, improve
4. **Fine-tuning** — Adapt to real robot specifics
5. **Patience** — Multiple iterations required (typical: 2-4 weeks)

Companies that excel at robotics (Boston Dynamics, DeepMind, Tesla humanoid projects) invest 50% of effort in **simulation and transfer**, not just in algorithms.

---

## Further Reading

- **Tobin et al. (2017): "Domain Randomization for Transferring Deep Neural Networks from Simulation to Real World"** — Foundational paper on randomization
- **Plappert et al. (2018): "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"** — Applied to robotic manipulation
- **James et al. (2018): "Sim-to-Real Robot Learning from Pixels with Progressive Nets"** — Transfer for vision-based control
- **OpenAI Dactyl Project (2019)** — Real-world robotic hand manipulation, extreme domain randomization
- **Prakash et al. (2021): "Towards Generalist Robots via Foundation Models"** — Vision-language models for robotics

---

**Next**: [Isaac Workflows](isaac-sim.md) — When simulation needs to be photorealistic. Learn about NVIDIA Isaac, synthetic data generation, and digital twins.

You now understand the complete pipeline for deploying simulation-trained policies to real robots. This is the core technology enabling the physical AI revolution.
