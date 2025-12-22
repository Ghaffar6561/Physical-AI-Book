---
sidebar_position: 6
title: "Bipedal Locomotion"
---

# Bipedal Locomotion and Balance Control

## Overview

Bipedal locomotion is one of the most challenging problems in robotics. Unlike wheeled robots, humanoids must constantly maintain dynamic balance while walking—a process that involves coordinating dozens of joints through carefully timed phases of support and swing.

## The Walking Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                    Bipedal Gait Cycle                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   0%        25%        50%        75%        100%           │
│   │          │          │          │          │             │
│   ▼          ▼          ▼          ▼          ▼             │
│  ┌─┐        ┌─┐        ┌─┐        ┌─┐        ┌─┐            │
│  │R│ ───▶   │R│ ───▶   │R│ ───▶   │R│ ───▶   │R│            │
│  └┬┘        └┬┘        └┬┘        └┬┘        └┬┘            │
│   │          │\         │          /│         │             │
│  ┌┴┐        ┌┴┐\       ┌┴┐       /┌┴┐        ┌┴┐            │
│  │L│        │L│ ▶      │L│      ◀ │L│        │L│            │
│  └─┘        └─┘        └─┘        └─┘        └─┘            │
│                                                              │
│  Double    Right      Double     Left       Double          │
│  Support   Swing      Support    Swing      Support         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Gait Phases

| Phase | Duration | Description |
|-------|----------|-------------|
| **Double Support** | 20% | Both feet on ground, weight transfer |
| **Single Support (R)** | 30% | Right foot stance, left foot swing |
| **Double Support** | 20% | Both feet on ground, weight transfer |
| **Single Support (L)** | 30% | Left foot stance, right foot swing |

## Zero Moment Point (ZMP) Control

ZMP is the point where the total inertial forces equal zero—the key to dynamic balance:

```python
import numpy as np

class ZMPController:
    """ZMP-based walking controller."""

    def __init__(self, robot_height=1.0, gravity=9.81):
        self.h = robot_height  # CoM height
        self.g = gravity
        self.omega = np.sqrt(self.g / self.h)  # Natural frequency

    def compute_zmp_from_com(self, com_pos, com_acc):
        """Compute ZMP from CoM position and acceleration."""
        zmp_x = com_pos[0] - (com_pos[2] / self.g) * com_acc[0]
        zmp_y = com_pos[1] - (com_pos[2] / self.g) * com_acc[1]
        return np.array([zmp_x, zmp_y])

    def generate_zmp_trajectory(self, footsteps, step_duration):
        """Generate ZMP reference trajectory from footsteps."""
        zmp_trajectory = []

        for i, footstep in enumerate(footsteps):
            # ZMP moves to center of support foot
            support_pos = np.array([footstep.x, footstep.y])

            # During double support, ZMP transitions between feet
            if i < len(footsteps) - 1:
                next_pos = np.array([footsteps[i+1].x, footsteps[i+1].y])

                # Single support phase
                for t in np.linspace(0, step_duration * 0.8, 20):
                    zmp_trajectory.append(support_pos)

                # Double support transition
                for t in np.linspace(0, 1, 5):
                    zmp_trajectory.append(support_pos + t * (next_pos - support_pos))

        return np.array(zmp_trajectory)

    def preview_control(self, zmp_ref, horizon=1.0, dt=0.01):
        """
        Preview control for CoM trajectory generation.
        Computes optimal CoM trajectory that tracks ZMP reference.
        """
        N = int(horizon / dt)

        # State-space model for Linear Inverted Pendulum
        # x = [com_x, com_vel_x, com_acc_x]
        A = np.array([
            [1, dt, dt**2/2],
            [0, 1, dt],
            [0, 0, 1]
        ])

        B = np.array([[dt**3/6], [dt**2/2], [dt]])

        C = np.array([[1, 0, -self.h/self.g]])

        # Preview gains (pre-computed for efficiency)
        Q = 1.0  # ZMP tracking weight
        R = 1e-6  # Input weight

        # Solve Riccati equation (simplified)
        # ... LQR/preview control implementation

        return com_trajectory
```

## Linear Inverted Pendulum Model (LIPM)

The LIPM simplifies humanoid dynamics for real-time control:

```python
class LIPMController:
    """Linear Inverted Pendulum Model for walking."""

    def __init__(self, com_height=0.9, step_time=0.5):
        self.h = com_height
        self.T = step_time
        self.g = 9.81
        self.omega = np.sqrt(self.g / self.h)

    def compute_com_trajectory(self, initial_com, initial_vel, target_zmp):
        """
        Analytical solution for CoM trajectory given ZMP.

        LIPM equation: x'' = omega^2 * (x - p)
        where x is CoM position, p is ZMP
        """
        x0 = initial_com - target_zmp
        v0 = initial_vel

        def com_position(t):
            return (target_zmp +
                    x0 * np.cosh(self.omega * t) +
                    v0 / self.omega * np.sinh(self.omega * t))

        def com_velocity(t):
            return (x0 * self.omega * np.sinh(self.omega * t) +
                    v0 * np.cosh(self.omega * t))

        return com_position, com_velocity

    def capture_point(self, com_pos, com_vel):
        """
        Compute capture point (DCM - Divergent Component of Motion).
        If ZMP is placed at capture point, robot will come to rest.
        """
        return com_pos + com_vel / self.omega

    def is_capturable(self, capture_point, support_polygon):
        """Check if robot can recover balance."""
        return self.point_in_polygon(capture_point, support_polygon)
```

## Swing Leg Trajectory

```python
class SwingLegController:
    """Generates smooth swing leg trajectories."""

    def __init__(self):
        self.swing_height = 0.05  # 5cm foot lift
        self.swing_duration = 0.4  # seconds

    def generate_trajectory(self, start_pos, end_pos, t):
        """
        Generate swing foot trajectory using cubic spline.
        t: normalized time [0, 1]
        """
        # Horizontal: smooth interpolation
        x = start_pos[0] + (end_pos[0] - start_pos[0]) * self.smooth_step(t)
        y = start_pos[1] + (end_pos[1] - start_pos[1]) * self.smooth_step(t)

        # Vertical: parabolic arc
        z = start_pos[2] + 4 * self.swing_height * t * (1 - t)

        # Add end position z
        z += (end_pos[2] - start_pos[2]) * t

        return np.array([x, y, z])

    def smooth_step(self, t):
        """Smooth step function (zero velocity at endpoints)."""
        return 3 * t**2 - 2 * t**3

    def generate_velocity(self, start_pos, end_pos, t):
        """Compute swing foot velocity."""
        # Derivative of position trajectory
        dx = (end_pos[0] - start_pos[0]) * 6 * t * (1 - t) / self.swing_duration
        dy = (end_pos[1] - start_pos[1]) * 6 * t * (1 - t) / self.swing_duration
        dz = 4 * self.swing_height * (1 - 2*t) / self.swing_duration

        return np.array([dx, dy, dz])
```

## Full Walking Controller

```python
class WalkingController:
    """Complete walking controller combining all components."""

    def __init__(self, robot):
        self.robot = robot
        self.lipm = LIPMController()
        self.swing = SwingLegController()
        self.state = 'standing'
        self.phase = 0.0  # [0, 1] within current step

    def step(self, dt):
        """Main control loop - call at 100Hz+."""
        if self.state == 'walking':
            self.phase += dt / self.step_time

            if self.phase >= 1.0:
                self.phase = 0.0
                self.switch_support_foot()

            # Compute targets
            com_target = self.compute_com_target()
            swing_target = self.compute_swing_target()

            # Inverse kinematics
            joint_angles = self.compute_joint_angles(com_target, swing_target)

            return joint_angles

        return self.robot.standing_pose

    def compute_com_target(self):
        """Compute CoM position for current phase."""
        com_func, _ = self.lipm.compute_com_trajectory(
            self.current_com,
            self.current_com_vel,
            self.current_zmp
        )
        return com_func(self.phase * self.step_time)

    def compute_swing_target(self):
        """Compute swing foot position for current phase."""
        return self.swing.generate_trajectory(
            self.swing_start,
            self.swing_end,
            self.phase
        )

    def compute_joint_angles(self, com_target, swing_target):
        """Convert Cartesian targets to joint angles."""
        # Support leg: maintain CoM position
        support_angles = self.robot.ik_leg(
            self.support_foot,
            com_target,
            self.support_foot_pos
        )

        # Swing leg: track foot trajectory
        swing_angles = self.robot.ik_leg(
            self.swing_foot,
            com_target,
            swing_target
        )

        return {**support_angles, **swing_angles}
```

## Balance Recovery

```python
class BalanceRecovery:
    """Reactive balance recovery strategies."""

    def __init__(self, robot):
        self.robot = robot
        self.push_threshold = 0.1  # m/s CoM velocity

    def detect_disturbance(self, com_vel, expected_vel):
        """Detect external push or slip."""
        error = np.linalg.norm(com_vel - expected_vel)
        return error > self.push_threshold

    def ankle_strategy(self, com_error):
        """
        Ankle strategy: small disturbances.
        Adjust ankle torque to shift CoP.
        """
        # Proportional control on ankle joints
        ankle_correction = -self.Kp_ankle * com_error
        return {'ankle_pitch': ankle_correction[0],
                'ankle_roll': ankle_correction[1]}

    def hip_strategy(self, com_vel_error):
        """
        Hip strategy: medium disturbances.
        Rotate upper body to generate angular momentum.
        """
        # Accelerate torso to counteract
        hip_correction = -self.Kp_hip * com_vel_error
        return {'torso_pitch': hip_correction[0],
                'torso_roll': hip_correction[1]}

    def stepping_strategy(self, capture_point, support_polygon):
        """
        Stepping strategy: large disturbances.
        Take a step to place foot under capture point.
        """
        if not self.is_capturable(capture_point, support_polygon):
            # Emergency step required
            step_target = capture_point + np.array([0.1, 0, 0])  # Overshoot
            return self.generate_emergency_step(step_target)

        return None
```

## Key Takeaways

| Concept | Purpose |
|---------|---------|
| **ZMP** | Dynamic balance criterion for walking |
| **LIPM** | Simplified model for real-time control |
| **Preview Control** | Anticipatory CoM trajectory planning |
| **Capture Point** | Balance recoverability metric |
| **Gait Phases** | Swing and stance coordination |
| **Recovery Strategies** | Ankle → Hip → Stepping |

---

*Next: Learn manipulation and grasping with humanoid hands.*
