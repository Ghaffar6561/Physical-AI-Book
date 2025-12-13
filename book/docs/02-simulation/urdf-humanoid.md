# Humanoid Robot Description with URDF

## What is URDF?

URDF (Unified Robot Description Format) is an XML format that describes:
- Links (body segments: torso, arms, legs)
- Joints (connections between links: revolute, prismatic)
- Sensors (cameras, LiDAR, IMU attached to links)
- Physical properties (mass, inertia, collisions)

## URDF Example: Simple Humanoid

A minimal URDF for a humanoid includes:

```xml
<robot name="humanoid_robot">
  <!-- Torso -->
  <link name="torso">
    <inertial>...</inertial>
    <collision>...</collision>
    <visual>...</visual>
  </link>
  
  <!-- Right Arm -->
  <link name="right_shoulder">...</link>
  <joint name="right_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="right_shoulder"/>
  </joint>
  
  <!-- Left Arm, Legs, Head, etc. -->
  ...
</robot>
```

## Humanoid Structure

A typical humanoid has:
- **Torso** — Central body, usually 3 DOF (pitch, roll, yaw)
- **Head** — 2 DOF (pan, tilt) for camera orientation
- **Arms** — 7 DOF each (shoulder, elbow, wrist)
- **Legs** — 6 DOF each (hip, knee, ankle)
- **Gripper** — 1-5 DOF depending on complexity

Total: ~30-40 DOF for a humanoid.

## Building Your Own URDF

To come in exercises.

---

**Next**: [Setup Gazebo](setup-gazebo.md)
