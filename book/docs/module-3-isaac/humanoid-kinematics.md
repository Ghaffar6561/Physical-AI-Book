---
sidebar_position: 5
title: "Humanoid Kinematics"
---

# Humanoid Robot Kinematics and Dynamics

## Overview

**Kinematics** describes the motion of a robot without considering forces, while **dynamics** accounts for the forces and torques that cause motion. For humanoid robots, understanding both is essential for balance, locomotion, and manipulation.

## Humanoid Kinematic Structure

A typical humanoid has 30-40 degrees of freedom (DOF):

```
┌─────────────────────────────────────────────────────────────┐
│                 Humanoid Joint Structure                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                      ┌─────────┐                            │
│                      │  Head   │ (2 DOF: pan, tilt)         │
│                      └────┬────┘                            │
│                           │                                  │
│              ┌────────────┼────────────┐                    │
│              │            │            │                    │
│         ┌────▼────┐  ┌────▼────┐  ┌────▼────┐              │
│         │  L Arm  │  │  Torso  │  │  R Arm  │              │
│         │ (7 DOF) │  │ (3 DOF) │  │ (7 DOF) │              │
│         └────┬────┘  └────┬────┘  └────┬────┘              │
│              │            │            │                    │
│         ┌────▼────┐       │       ┌────▼────┐              │
│         │ L Hand  │       │       │ R Hand  │              │
│         │ (5 DOF) │       │       │ (5 DOF) │              │
│         └─────────┘       │       └─────────┘              │
│                           │                                  │
│              ┌────────────┴────────────┐                    │
│              │                         │                    │
│         ┌────▼────┐               ┌────▼────┐              │
│         │  L Leg  │               │  R Leg  │              │
│         │ (6 DOF) │               │ (6 DOF) │              │
│         └────┬────┘               └────┬────┘              │
│              │                         │                    │
│         ┌────▼────┐               ┌────▼────┐              │
│         │ L Foot  │               │ R Foot  │              │
│         └─────────┘               └─────────┘              │
│                                                              │
│         Total: ~37 DOF                                      │
└─────────────────────────────────────────────────────────────┘
```

## Joint Types and Limits

| Body Part | DOF | Joints | Typical Range |
|-----------|-----|--------|---------------|
| **Neck** | 2 | Pan, Tilt | ±60°, ±45° |
| **Torso** | 3 | Roll, Pitch, Yaw | ±30°, ±30°, ±60° |
| **Shoulder** | 3 | Roll, Pitch, Yaw | ±180°, ±90°, ±90° |
| **Elbow** | 1 | Pitch | 0° to 150° |
| **Wrist** | 3 | Roll, Pitch, Yaw | ±90°, ±70°, ±90° |
| **Hip** | 3 | Roll, Pitch, Yaw | ±45°, ±120°, ±45° |
| **Knee** | 1 | Pitch | 0° to 150° |
| **Ankle** | 2 | Roll, Pitch | ±30°, ±45° |

## Forward Kinematics

Forward kinematics computes the end-effector position given joint angles:

```python
import numpy as np
from scipy.spatial.transform import Rotation

class HumanoidFK:
    """Forward kinematics for humanoid robot."""

    def __init__(self, urdf_path):
        self.chain = self.load_kinematic_chain(urdf_path)

    def compute_transform(self, joint_name, parent_transform, joint_angle):
        """Compute transform for a single joint."""
        joint = self.chain[joint_name]

        # Joint rotation (around joint axis)
        if joint.type == 'revolute':
            R = Rotation.from_rotvec(joint.axis * joint_angle)
        else:
            R = Rotation.identity()

        # Full transform: parent * joint_origin * rotation
        joint_transform = np.eye(4)
        joint_transform[:3, :3] = R.as_matrix()
        joint_transform[:3, 3] = joint.origin

        return parent_transform @ joint_transform

    def forward_kinematics(self, joint_angles, end_effector='right_hand'):
        """Compute end-effector pose from joint angles."""
        current_transform = np.eye(4)

        # Traverse kinematic chain from base to end-effector
        for joint_name in self.get_chain_to(end_effector):
            angle = joint_angles.get(joint_name, 0.0)
            current_transform = self.compute_transform(
                joint_name, current_transform, angle
            )

        position = current_transform[:3, 3]
        rotation = Rotation.from_matrix(current_transform[:3, :3])

        return position, rotation


# Usage
fk = HumanoidFK('/models/humanoid.urdf')

joint_angles = {
    'right_shoulder_pitch': 0.5,
    'right_shoulder_roll': 0.2,
    'right_shoulder_yaw': 0.0,
    'right_elbow_pitch': 1.2,
    'right_wrist_roll': 0.0,
    'right_wrist_pitch': 0.0,
}

hand_position, hand_rotation = fk.forward_kinematics(joint_angles, 'right_hand')
print(f"Hand position: {hand_position}")
```

## Inverse Kinematics (IK)

Inverse kinematics computes joint angles to achieve a desired end-effector pose:

```python
from scipy.optimize import minimize

class HumanoidIK:
    """Inverse kinematics solver for humanoid robot."""

    def __init__(self, fk_solver):
        self.fk = fk_solver

    def ik_cost(self, joint_angles_flat, target_pos, target_rot, joint_names):
        """Cost function for IK optimization."""
        # Reconstruct joint dict
        joint_angles = dict(zip(joint_names, joint_angles_flat))

        # Compute current end-effector pose
        current_pos, current_rot = self.fk.forward_kinematics(joint_angles)

        # Position error
        pos_error = np.linalg.norm(target_pos - current_pos)

        # Orientation error (geodesic distance)
        rot_error = (target_rot.inv() * current_rot).magnitude()

        return pos_error + 0.5 * rot_error

    def solve(self, target_pos, target_rot, initial_guess, end_effector='right_hand'):
        """Solve IK for target pose."""
        joint_names = list(initial_guess.keys())
        initial_angles = list(initial_guess.values())

        # Get joint limits
        bounds = self.get_joint_bounds(joint_names)

        result = minimize(
            self.ik_cost,
            initial_angles,
            args=(target_pos, target_rot, joint_names),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100}
        )

        if result.success:
            return dict(zip(joint_names, result.x))
        else:
            raise ValueError("IK solution not found")

    def solve_analytical_leg(self, foot_target, hip_position):
        """
        Analytical IK for 6-DOF leg.
        Faster than numerical for known geometries.
        """
        # Leg dimensions
        thigh_length = 0.40  # meters
        shin_length = 0.40   # meters

        # Vector from hip to foot target
        hip_to_foot = foot_target - hip_position
        distance = np.linalg.norm(hip_to_foot)

        # Check reachability
        if distance > thigh_length + shin_length:
            raise ValueError("Target out of reach")

        # Compute knee angle using law of cosines
        cos_knee = (thigh_length**2 + shin_length**2 - distance**2) / \
                   (2 * thigh_length * shin_length)
        knee_angle = np.arccos(np.clip(cos_knee, -1, 1))

        # Compute hip angles
        # ... geometric calculations

        return {
            'hip_pitch': hip_pitch,
            'hip_roll': hip_roll,
            'hip_yaw': hip_yaw,
            'knee_pitch': knee_angle,
            'ankle_pitch': ankle_pitch,
            'ankle_roll': ankle_roll,
        }
```

## Dynamics: Equations of Motion

Humanoid dynamics follow the manipulator equation:

```
M(q)q̈ + C(q,q̇)q̇ + G(q) = τ + Jᵀf_ext
```

Where:
- **M(q)**: Mass/inertia matrix
- **C(q,q̇)**: Coriolis/centrifugal terms
- **G(q)**: Gravity vector
- **τ**: Joint torques
- **J**: Jacobian
- **f_ext**: External forces (contact)

```python
import pinocchio as pin

class HumanoidDynamics:
    """Dynamics computation using Pinocchio."""

    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

    def compute_dynamics(self, q, v, a):
        """Compute inverse dynamics: τ = M*a + C*v + G"""
        # Joint configuration
        q_pin = np.array(q)
        v_pin = np.array(v)  # velocities
        a_pin = np.array(a)  # accelerations

        # Inverse dynamics
        tau = pin.rnea(self.model, self.data, q_pin, v_pin, a_pin)

        return tau

    def compute_mass_matrix(self, q):
        """Compute mass/inertia matrix M(q)."""
        pin.crba(self.model, self.data, np.array(q))
        return self.data.M

    def compute_gravity(self, q):
        """Compute gravity vector G(q)."""
        return pin.computeGeneralizedGravity(self.model, self.data, np.array(q))

    def compute_com(self, q):
        """Compute center of mass position."""
        pin.centerOfMass(self.model, self.data, np.array(q))
        return self.data.com[0]

    def compute_zmp(self, q, v, a):
        """Compute Zero Moment Point for balance."""
        # Compute total force and moment at CoM
        pin.forwardKinematics(self.model, self.data, q, v, a)

        com = self.compute_com(q)
        total_mass = sum([self.model.inertias[i].mass
                          for i in range(self.model.njoints)])

        # ZMP calculation
        gravity = 9.81
        com_acc = self.data.acom[0]

        zmp_x = com[0] - com[2] * com_acc[0] / (gravity + com_acc[2])
        zmp_y = com[1] - com[2] * com_acc[1] / (gravity + com_acc[2])

        return np.array([zmp_x, zmp_y, 0.0])
```

## Center of Mass and Balance

```python
class BalanceController:
    """Balance control using CoM and ZMP."""

    def __init__(self, dynamics):
        self.dynamics = dynamics

        # Support polygon (foot positions)
        self.foot_width = 0.10
        self.foot_length = 0.20

    def is_balanced(self, q, v, a, stance_foot):
        """Check if ZMP is within support polygon."""
        zmp = self.dynamics.compute_zmp(q, v, a)

        # Define support polygon based on stance
        if stance_foot == 'double':
            # Both feet on ground - larger polygon
            support = self.get_double_support_polygon()
        else:
            # Single foot support
            support = self.get_single_support_polygon(stance_foot)

        return self.point_in_polygon(zmp[:2], support)

    def compute_balance_correction(self, q, v, target_com):
        """Compute joint corrections to maintain balance."""
        current_com = self.dynamics.compute_com(q)
        com_error = target_com - current_com

        # Use CoM Jacobian to compute joint corrections
        J_com = self.compute_com_jacobian(q)

        # Pseudo-inverse for redundant system
        J_pinv = np.linalg.pinv(J_com)
        dq = J_pinv @ com_error

        return dq
```

## Jacobian for Manipulation

```python
def compute_end_effector_jacobian(self, q, end_effector):
    """Compute Jacobian for end-effector velocity control."""
    frame_id = self.model.getFrameId(end_effector)

    pin.computeJointJacobians(self.model, self.data, q)
    J = pin.getFrameJacobian(
        self.model, self.data, frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    return J

def velocity_control(self, q, desired_velocity, end_effector):
    """Compute joint velocities for desired end-effector velocity."""
    J = self.compute_end_effector_jacobian(q, end_effector)

    # Damped least squares (more stable than pseudo-inverse)
    damping = 0.01
    J_damped = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))

    dq = J_damped @ desired_velocity

    return dq
```

## Key Takeaways

| Concept | Purpose |
|---------|---------|
| **Forward Kinematics** | Joint angles → End-effector pose |
| **Inverse Kinematics** | Desired pose → Joint angles |
| **Dynamics** | Forces/torques ↔ Accelerations |
| **Center of Mass** | Balance and stability |
| **ZMP** | Dynamic balance criterion |
| **Jacobian** | Velocity/force transformation |

---

*Next: Learn bipedal locomotion and balance control.*
