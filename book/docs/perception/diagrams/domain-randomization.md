# Domain Randomization Visualization

This document describes the domain randomization process and its effect on policy robustness.

## Core Concept: Randomization Builds Robustness

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              DOMAIN RANDOMIZATION: FROM OVERFITTING TO ROBUSTNESS            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DETERMINISTIC TRAINING (No Randomization)                                 │
│  ═══════════════════════════════════════════════════════════════════       │
│                                                                             │
│  Episode 1:  μ = 0.7  ✓  Position exact  ✓  Lighting fixed  ✓             │
│  Episode 2:  μ = 0.7  ✓  Position exact  ✓  Lighting fixed  ✓             │
│  Episode 3:  μ = 0.7  ✓  Position exact  ✓  Lighting fixed  ✓             │
│  ...                                                                        │
│                                                                             │
│  Policy learns: "Always use F=50N gripper force"                           │
│                 "Always approach from angle=0°"                            │
│                 "Lighting is always from north"                            │
│                                                                             │
│  Result: 98% success in SIM ✓                                              │
│          30% success in REAL ✗ (different friction, lighting, etc.)        │
│                                                                             │
│                                                                             │
│  RANDOMIZED TRAINING (Domain Randomization)                                │
│  ═════════════════════════════════════════════════════════════════════     │
│                                                                             │
│  Episode 1:  μ = 0.45  ✓  Position ±3cm  ✓  Light from NW  ✓             │
│  Episode 2:  μ = 0.82  ✓  Position ±4cm  ✓  Light from SE  ✓             │
│  Episode 3:  μ = 0.31  ✓  Position ±2cm  ✓  Light overhead ✓             │
│  Episode 4:  μ = 0.91  ✓  Position ±5cm  ✓  Light from SW  ✓             │
│  ...                                                                        │
│                                                                             │
│  Policy learns: "Adapt gripper force based on feedback"                    │
│                 "Work from multiple angles"                                │
│                 "Vision works with any lighting"                           │
│                                                                             │
│  Result: 85% success in SIM ✓                                              │
│          85% success in REAL ✓ (generalizes to new conditions!)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Randomization Design Space

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           WHAT TO RANDOMIZE: Parameter Selection & Ranges                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CRITICAL (Always Include):                                                │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  Parameter              Default    Range          Real-World Match         │
│  ────────────────────────────────────────────────────────────────────      │
│  Object Friction        μ = 0.7    [0.3, 1.0]     Covers glass→rubber      │
│  Object Mass            m = 1.0 kg [0.8, 1.2] kg  ±20% manufacturing       │
│  Object Position        (0.5, y)   ±0.05 m        Placement tolerance      │
│  Motor Delay            0 ms       [0, 50] ms     Real motor latency       │
│  Camera Noise           σ = 0      σ ∈ [0, 0.05]  Real sensor noise        │
│  Lighting Direction     North      Random sphere  Time of day variation     │
│  Lighting Intensity     1.0        [0.5, 1.5]     Sunny vs cloudy          │
│                                                                             │
│  OPTIONAL (Helps Robustness):                                              │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  Object Color           Red        RGB random     Paint variation          │
│  Gravity                9.81       [9.70, 9.85]   Geographic variation     │
│  Table Friction         0.8        [0.3, 0.9]     Surface wear             │
│  Camera Exposure        Auto       [±20%]         Sensor auto-adjust       │
│  Contact Model          ODE        Bullet/PhysX   Physics engine variation  │
│                                                                             │
│  NOT RECOMMENDED (Breaks Learning):                                        │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│  Robot Arm Length       1.0 m      Random         Fundamental task change  │
│  Number of Joints       30         Random         Architecture change      │
│  Task Objective         Grasp      Random         Defeats learning         │
│  Physics Engine         ODE        Random switch  Inconsistent rewards     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Training Progression with Randomization

```
Policy Performance Over Training Episodes

Success Rate (%)
      │
  100 │
      │          ┌────────────────────────────── Deterministic baseline
   90 │         │                                 (overfits to sim)
      │         │
   80 │         │    ┌─────────────────────────── Randomized policy
      │         │   │                             (generalizes)
   70 │         │   │
      │         │   │
   60 │         │   │
      │         │   │
   50 │         │   │
      │  ╱─────╱    │
   40 │╱╱           │
      │             │
   30 │             │
      │             │
   20 │             │
      │             │
   10 │             │
      │             │
    0 │_____________│_________________________________
      0           500          1000       1500    Episodes

Key observation:
- Deterministic: Rapid rise to 98%, but performance drops on real robot
- Randomized: Slower rise to 85%, but transfers to real robot at 85%!

Learning efficiency trade-off:
- Randomized takes ~20% more training episodes
- But avoids catastrophic failure on real hardware
- Net win for robotics applications
```

## Real-World Success Rates by Randomization Level

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            REAL-WORLD PERFORMANCE vs RANDOMIZATION LEVEL                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Randomization Level        Sim Success    Real Success    Transfer Ratio  │
│  ────────────────────────────────────────────────────────────────────────  │
│  None (baseline)            98%            30%             0.31 ✗          │
│  Light (±10% variation)     95%            45%             0.47 ✗          │
│  Medium (±20% variation)    90%            65%             0.72 ✓          │
│  Heavy (match real range)   85%            85%             1.00 ✓✓         │
│  Extreme (beyond real)      70%            82%             1.17 ✓✓         │
│                                                                             │
│  Recommendation: Use "Heavy" randomization (match real-world variation)     │
│  Extreme is overkill unless transfer still failing                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Randomization Process in Training Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           TRAINING LOOP WITH DOMAIN RANDOMIZATION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  for episode in range(1000):                                               │
│      ┌─────────────────────────────────────────────────────┐               │
│      │ Step 1: Sample Random Parameters                    │               │
│      ├─────────────────────────────────────────────────────┤               │
│      │ friction = uniform(0.3, 1.0)                        │               │
│      │ obj_mass = normal(1.0, 0.1)  # ±10%                │               │
│      │ obj_pos = nominal + normal(0, 0.05)  # ±5cm        │               │
│      │ light_dir = random_unit_vector()                    │               │
│      │ camera_noise = uniform(0, 0.05)                     │               │
│      │ motor_delay = uniform(0, 50) ms                     │               │
│      │                                                     │               │
│      │ World parameters: {friction, mass, pos, ...}       │               │
│      └─────────────────────────────────────────────────────┘               │
│                         ↓                                                   │
│      ┌─────────────────────────────────────────────────────┐               │
│      │ Step 2: Update Gazebo Simulation                    │               │
│      ├─────────────────────────────────────────────────────┤               │
│      │ update_gazebo_world(parameters)                     │               │
│      │ Set object friction coefficient                     │               │
│      │ Set object mass                                     │               │
│      │ Set initial pose                                    │               │
│      │ Set lighting in SDF world file                      │               │
│      │ Inject sensor noise plugin                          │               │
│      │ Set motor response model                            │               │
│      └─────────────────────────────────────────────────────┘               │
│                         ↓                                                   │
│      ┌─────────────────────────────────────────────────────┐               │
│      │ Step 3: Run Episode in Randomized World             │               │
│      ├─────────────────────────────────────────────────────┤               │
│      │ reset_robot_to_home()                               │               │
│      │ for t in range(100):  # 100 timesteps per episode   │               │
│      │     observation = perceive()                        │               │
│      │     action = policy(observation)                    │               │
│      │     state, reward, done = step(action)              │               │
│      │     store_transition(obs, action, reward)           │               │
│      │                                                     │               │
│      │ Result: Trajectory in THIS randomized world         │               │
│      └─────────────────────────────────────────────────────┘               │
│                         ↓                                                   │
│      ┌─────────────────────────────────────────────────────┐               │
│      │ Step 4: Update Policy with Trajectory              │               │
│      ├─────────────────────────────────────────────────────┤               │
│      │ policy.update(stored_transitions)                   │               │
│      │                                                     │               │
│      │ Important: Policy never sees same parameters twice!  │               │
│      │ Each episode teaches: "handle variation"            │               │
│      └─────────────────────────────────────────────────────┘               │
│                         ↓                                                   │
│      After 1000 episodes, policy has seen:                 │               │
│      - 1000 different friction values                      │               │
│      - 1000 different object positions                     │               │
│      - 1000 different lighting conditions                  │               │
│      - 1000 different motor responses                      │               │
│                                                                             │
│      Result: Policy robust to variation (like real robot!)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Success Stories: Domain Randomization in Practice

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 REAL ROBOTS: DOMAIN RANDOMIZATION SUCCESS                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OpenAI Robotic Hand (Dactyl) - 2018                                       │
│  ─────────────────────────────────────────────────────────────────────     │
│  Task: Complex hand manipulation (rotating cube, writing, etc.)            │
│                                                                             │
│  Without randomization:                                                    │
│    Sim success: 95%   →   Real success: 10%  ✗                             │
│                                                                             │
│  With domain randomization (extreme):                                       │
│    - Random friction [0.1, 2.0] (4× range)                                 │
│    - Random object size ±50%                                               │
│    - Random hand properties (friction, mass)                               │
│    - Random lighting (complete spectrum)                                    │
│    - Camera noise, motion blur, latency                                    │
│                                                                             │
│    Sim success: 80%   →   Real success: 76%  ✓✓ (Transfer ratio: 0.95!)    │
│                                                                             │
│  Key: "Extreme randomization" made sim-to-real transfer nearly perfect!     │
│                                                                             │
│                                                                             │
│  UC Berkeley Robotic Grasping - 2017                                       │
│  ─────────────────────────────────────────────────────────────────────     │
│  Task: Grasp unknown objects from bin with suction gripper                 │
│                                                                             │
│  Baseline (no randomization):                                              │
│    Sim: 92%   →   Real: 18%  ✗                                             │
│                                                                             │
│  With randomization:                                                       │
│    - Object texture, size, color                                           │
│    - Suction cup wear (surface roughness)                                  │
│    - Lighting conditions                                                   │
│    - Table clutter (density of objects)                                    │
│    - Camera viewpoint (multiple angles)                                    │
│                                                                             │
│    Sim: 85%   →   Real: 70%  ✓  (4× improvement over baseline!)            │
│                                                                             │
│  Key: Randomization particularly helps vision-based grasping               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## When Randomization Isn't Enough

```
Scenario 1: INSUFFICIENT RANDOMIZATION
─────────────────────────────────────
  Randomization range: [0.5, 0.9]  (narrow)
  Real range: [0.2, 1.2]  (wide)

  Result: "Sim covers subset of reality"
          Policy unprepared for extreme cases

  Fix: Expand randomization to match real range
       Or use hardware-in-the-loop to discover correct range


Scenario 2: RANDOMIZATION TOO EXTREME
──────────────────────────────────────
  Randomization: [0.0, 2.0]  (extreme)
  Real range: [0.2, 0.8]  (normal)

  Result: Policy overtrained for useless edge cases
          Learning is slow (task too hard)
          Sim success: 40% (policy never converges)

  Fix: Reduce to realistic ranges
       Match real-world distribution, not extremes


Scenario 3: RANDOMIZATION ON WRONG PARAMETERS
──────────────────────────────────────────────
  Randomized: Arm length (5%-20% variation)
  Real: Arm length constant

  Result: Policy learns to work with variable morphology
          Unnecessarily slow convergence
          Confusion on real robot (arm length is stable)

  Fix: Only randomize parameters that actually vary
       Keep physics fundamentals stable
```

---

## Design Checklist for Domain Randomization

```
Before training, verify:

□ Critical parameters randomized
  □ Object properties (friction, mass, size)
  □ Initial pose (position ±tolerance, orientation)
  □ Lighting (direction, intensity)
  □ Camera (noise, latency, distortion)
  □ Motor (delay, response characteristics)

□ Randomization ranges realistic
  □ Match or slightly exceed real-world variation
  □ Document why each range was chosen
  □ Validate ranges against real data

□ Non-varied parameters documented
  □ List what is NOT randomized
  □ Explain why (e.g., "arm length is fixed")
  □ Flag if these assumptions might break

□ Task remains learnable
  □ Baseline policy reaches >70% in sim
  □ If not, reduce randomization

□ Convergence monitoring
  □ Track sim success over episodes
  □ Should plateau around 80-90%
  □ If stuck at 60%, randomization too extreme
```

---

## Key Insights

1. **Randomization trades peak performance for robustness**: 98% → 85% in sim, but 30% → 85% in real
2. **Heavy randomization outperforms light**: Match real-world variation magnitude
3. **Different tasks need different parameters**: Vision task randomizes cameras; control task randomizes physics
4. **Convergence is slower**: ~20% more training episodes, but essential for transfer
5. **Combined with HiL**: Randomization + hardware testing = systematic sim-to-real closure

---

## Further Study

- Tobin et al. "Domain Randomization for Transferring Deep Neural Networks from Simulation to Real World" (2017)
- Plappert et al. "Sim-to-Real: Learning Agile Locomotion For Quadruped Robots" (2018)
- OpenAI Robotic Hand white paper for extreme randomization examples
