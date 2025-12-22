# URDF: Describing Humanoid Robots

URDF (Unified Robot Description Format) is an XML language that describes robot structure, kinematics, inertias, and sensors. It's the standard format used across the ROS 2 ecosystem.

## URDF Basics

A URDF file contains:

1. **Links** — Rigid body segments (torso, arm, leg)
2. **Joints** — Connections between links (revolute/rotating, prismatic/sliding)
3. **Inertia** — Mass, moment of inertia (for physics simulation)
4. **Collision/Visual Geometry** — Shapes for collision detection and rendering
5. **Sensors** — Cameras, LiDAR, IMU attached to links

**Minimal URDF Structure:**

```xml
<?xml version="1.0"?>
<robot name="humanoid_simple">
  <!-- Link: Base body -->
  <link name="torso">
    <inertial>
      <mass value="10.0"/>  <!-- 10 kg -->
      <inertia ixx="0.5" iyy="0.5" izz="0.3" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.2 0.2 0.5"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.2 0.5"/>
      </geometry>
    </collision>
  </link>

  <!-- Link: Upper arm -->
  <link name="arm_upper">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" iyy="0.01" izz="0.05" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint: Connects torso to upper arm -->
  <joint name="shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="arm_upper"/>
    <origin xyz="0.1 0 0.25" rpy="0 0 0"/>  <!-- Position and orientation -->
    <axis xyz="0 1 0"/>  <!-- Rotate around Y axis -->
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1.57"/>
    <!-- Range: -90° to +90°, Max torque: 50 Nm, Max speed: 1.57 rad/s (90°/s) -->
    <dynamics damping="0.1" friction="0.1"/>
  </joint>
</robot>
```

## Humanoid Structure

A typical humanoid robot has:

| Body Part | DOF | Purpose |
|-----------|-----|---------|
| **Torso** | 0-3 | Waist pitch/roll/yaw (optional, depends on design) |
| **Neck** | 2 | Pan (yaw), tilt (pitch) for camera |
| **Left Arm** | 7 | Shoulder (3), elbow (1), wrist (3) |
| **Right Arm** | 7 | Same as left |
| **Left Leg** | 6 | Hip (3), knee (1), ankle (2) |
| **Right Leg** | 6 | Same as left |
| **Gripper** | 1-5 | Open/close, individual fingers (optional) |
| **Total** | 30-37 | Typical humanoid degrees of freedom |

## Building a Humanoid URDF

### Step 1: Define the Torso

The torso is the base of the kinematic chain. All other limbs attach to it.

```xml
<link name="torso">
  <inertial>
    <!-- Mass typical for humanoid: 10-20 kg for torso -->
    <mass value="15.0"/>  <!-- 15 kg -->
    <!-- Inertia: for a 0.2m x 0.2m x 0.5m box of 15 kg
         Ixx = m * (d² + h²) / 12 = 15 * (0.2² + 0.5²) / 12 ≈ 0.354
         Iyy = m * (w² + h²) / 12 = 15 * (0.2² + 0.5²) / 12 ≈ 0.354
         Izz = m * (w² + d²) / 12 = 15 * (0.2² + 0.2²) / 12 ≈ 0.1 -->
    <inertia ixx="0.354" iyy="0.354" izz="0.1" ixy="0" ixz="0" iyz="0"/>
  </inertial>
  <visual>
    <geometry>
      <box size="0.2 0.2 0.5"/>
    </geometry>
    <material name="torso_color">
      <color rgba="0.5 0.5 0.5 1.0"/>  <!-- Gray -->
    </material>
  </visual>
  <collision>
    <geometry>
      <box size="0.2 0.2 0.5"/>
    </geometry>
  </collision>
</link>
```

### Step 2: Add Arms

Each arm has shoulder (pitch, roll, yaw), elbow (pitch), and wrist (pitch, roll, yaw).

```xml
<!-- Right Shoulder Joint -->
<joint name="right_shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="right_arm_shoulder"/>
  <origin xyz="0.15 -0.1 0.2" rpy="0 0 0"/>  <!-- Attach to right side of torso -->
  <axis xyz="0 1 0"/>  <!-- Pitch: rotate around Y -->
  <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  <dynamics damping="0.2" friction="0.05"/>
</joint>

<link name="right_arm_shoulder">
  <inertial>
    <mass value="2.5"/>
    <inertia ixx="0.05" iyy="0.01" izz="0.05" ixy="0" ixz="0" iyz="0"/>
  </inertial>
  <visual>
    <geometry>
      <cylinder radius="0.035" length="0.15"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.035" length="0.15"/>
    </geometry>
  </collision>
</link>

<!-- Right Elbow Joint -->
<joint name="right_elbow_pitch" type="revolute">
  <parent link="right_arm_shoulder"/>
  <child link="right_arm_elbow"/>
  <origin xyz="0 0 -0.2" rpy="0 0 0"/>  <!-- Below shoulder -->
  <axis xyz="0 1 0"/>  <!-- Pitch: rotate around Y -->
  <limit lower="0" upper="2.5" effort="30" velocity="2.0"/>  <!-- 0-143° -->
  <dynamics damping="0.15" friction="0.05"/>
</joint>

<link name="right_arm_elbow">
  <inertial>
    <mass value="1.5"/>
    <inertia ixx="0.02" iyy="0.005" izz="0.02" ixy="0" ixz="0" iyz="0"/>
  </inertial>
  <visual>
    <geometry>
      <cylinder radius="0.03" length="0.25"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.03" length="0.25"/>
    </geometry>
  </collision>
</link>

<!-- Right Gripper/Hand -->
<joint name="right_gripper" type="prismatic">
  <parent link="right_arm_elbow"/>
  <child link="right_hand"/>
  <origin xyz="0 0 -0.15" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>  <!-- Slide along Z (open/close) -->
  <limit lower="0" upper="0.1" effort="100" velocity="1.0"/>  <!-- 0-10 cm opening -->
  <dynamics damping="0.5" friction="0.1"/>
</joint>

<link name="right_hand">
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
  </inertial>
  <visual>
    <geometry>
      <box size="0.08 0.08 0.15"/>  <!-- Hand size -->
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.08 0.08 0.15"/>
    </geometry>
  </collision>
</link>
```

*(Mirror this for left arm with opposite offsets)*

### Step 3: Add Legs

Each leg has hip (3 DOF), knee (1 DOF), and ankle (2 DOF).

```xml
<!-- Right Hip Pitch Joint -->
<joint name="right_hip_pitch" type="revolute">
  <parent link="torso"/>
  <child link="right_thigh"/>
  <origin xyz="0.05 -0.08 -0.25" rpy="0 0 0"/>  <!-- Below torso, right side -->
  <axis xyz="0 1 0"/>  <!-- Pitch: rotate around Y -->
  <limit lower="-1.57" upper="0.785" effort="100" velocity="1.5"/>  <!-- -90° to +45° -->
  <dynamics damping="0.3" friction="0.1"/>
</joint>

<link name="right_thigh">
  <inertial>
    <mass value="5.0"/>  <!-- Thigh is heavy -->
    <inertia ixx="0.1" iyy="0.02" izz="0.1" ixy="0" ixz="0" iyz="0"/>
  </inertial>
  <visual>
    <geometry>
      <cylinder radius="0.045" length="0.35"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.045" length="0.35"/>
    </geometry>
  </collision>
</link>

<!-- Right Knee Joint -->
<joint name="right_knee_pitch" type="revolute">
  <parent link="right_thigh"/>
  <child link="right_calf"/>
  <origin xyz="0 0 -0.35" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="0" upper="2.5" effort="80" velocity="1.5"/>  <!-- 0-143° -->
  <dynamics damping="0.25" friction="0.08"/>
</joint>

<link name="right_calf">
  <inertial>
    <mass value="3.0"/>
    <inertia ixx="0.05" iyy="0.01" izz="0.05" ixy="0" ixz="0" iyz="0"/>
  </inertial>
  <visual>
    <geometry>
      <cylinder radius="0.04" length="0.35"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.04" length="0.35"/>
    </geometry>
  </collision>
</link>

<!-- Right Ankle -->
<joint name="right_ankle_pitch" type="revolute">
  <parent link="right_calf"/>
  <child link="right_foot"/>
  <origin xyz="0 0 -0.35" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.785" upper="0.785" effort="50" velocity="2.0"/>  <!-- -45° to +45° -->
  <dynamics damping="0.2" friction="0.1"/>
</joint>

<link name="right_foot">
  <inertial>
    <mass value="1.5"/>
    <inertia ixx="0.01" iyy="0.005" izz="0.01" ixy="0" ixz="0" iyz="0"/>
  </inertial>
  <visual>
    <geometry>
      <box size="0.1 0.25 0.15"/>  <!-- Foot length × width × height -->
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.1 0.25 0.15"/>
    </geometry>
  </collision>
</link>
```

*(Mirror for left leg)*

### Step 4: Add a Head with Camera

```xml
<!-- Neck Joints -->
<joint name="neck_yaw" type="revolute">
  <parent link="torso"/>
  <child link="head"/>
  <origin xyz="0 0 0.3" rpy="0 0 0"/>  <!-- Top of torso -->
  <axis xyz="0 0 1"/>  <!-- Yaw: rotate around Z -->
  <limit lower="-1.57" upper="1.57" effort="20" velocity="2.0"/>
  <dynamics damping="0.1" friction="0.05"/>
</joint>

<link name="head">
  <inertial>
    <mass value="2.0"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
  </inertial>
  <visual>
    <geometry>
      <box size="0.15 0.15 0.2"/>  <!-- Head size -->
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.15 0.15 0.2"/>
    </geometry>
  </collision>
</link>

<!-- Camera attached to head -->
<joint name="camera_joint" type="fixed">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0 0 0.05" rpy="0 0 0"/>  <!-- Center of head, looking forward -->
</joint>

<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.05 0.05 0.05"/>  <!-- Camera size -->
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </collision>
</link>
```

## Key URDF Concepts

### Links

A link is a rigid body. It must have:
- **name**: Unique identifier
- **inertial**: Mass and inertia tensor
- **visual**: 3D shape for rendering
- **collision**: 3D shape for collision detection

```xml
<link name="example_link">
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.04" length="0.2"/>
    </geometry>
    <material name="material_name">
      <color rgba="1 0 0 1"/>  <!-- RGBA -->
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.04" length="0.2"/>
    </geometry>
  </collision>
</link>
```

### Joints

A joint connects two links. Types:

- **revolute**: Rotates around an axis (motors, hinges)
- **prismatic**: Slides along an axis (grippers, pistons)
- **fixed**: No movement (cameras, sensors rigidly attached)
- **floating**: Free movement (base frame)
- **planar**: 2D movement

```xml
<joint name="example_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>  <!-- Relative position -->
  <axis xyz="0 1 0"/>  <!-- Rotation axis (X, Y, or Z) -->
  <limit lower="-1.57" upper="1.57" effort="50" velocity="1.57"/>
  <!-- lower/upper: joint angle limits (radians) -->
  <!-- effort: max motor torque (Nm) -->
  <!-- velocity: max motor speed (rad/s) -->
  <dynamics damping="0.1" friction="0.05"/>
  <!-- damping: velocity-dependent friction -->
  <!-- friction: static friction/stiction -->
</joint>
```

### Inertia Tensor

The inertia tensor (Ixx, Iyy, Izz) describes resistance to rotation. For simple shapes:

**Box (width × depth × height):**
```
Ixx = m * (depth² + height²) / 12
Iyy = m * (width² + height²) / 12
Izz = m * (width² + depth²) / 12
```

**Cylinder (radius r, length L):**
```
Ixx = m * (3 * r² + L²) / 12  (around axis of rotation)
Iyy = Izz = m * r² / 2  (perpendicular axes)
```

For a humanoid, typical values:
- Torso: 0.3-0.5 kg⋅m²
- Arm segment: 0.01-0.05 kg⋅m²
- Leg segment: 0.05-0.15 kg⋅m²

### Coordinate Frames

Each link has a frame. The origin tag specifies position/orientation relative to parent:

```xml
<origin xyz="x y z" rpy="roll pitch yaw"/>
```

**Position (xyz)**: Meters relative to parent frame
**Rotation (rpy)**: Euler angles in radians
- Roll (φ): Rotation around X axis
- Pitch (θ): Rotation around Y axis
- Yaw (ψ): Rotation around Z axis

---

## Complete Humanoid Example

See `book/examples/humanoid-sim/gazebo_models/humanoid_simple.urdf` for a complete working example with:
- 30+ joints
- Torso, arms, legs, head
- Realistic masses and inertias
- Camera and IMU sensors
- Friction and damping parameters

## Validating Your URDF

Before using in Gazebo:

```bash
# Check URDF syntax
check_urdf your_robot.urdf

# Convert to graphical representation (dependency tree)
urdf_to_graphiz your_robot.urdf

# Load and inspect in RViz
rviz2 -d config.rviz
```

**Common URDF Errors:**
- Circular dependencies (joint chain loops)
- Negative inertia values (unstable)
- Link masses too small (unrealistic dynamics)
- Collision geometry doesn't match visual (floating appearance)

---

## Next Steps

1. Study the complete URDF example in `book/examples/`
2. Modify it to add new joints or sensors
3. Load it in Gazebo and verify behavior
4. Adjust masses and inertias based on real robot
5. (Advanced) Use URDF macros for repeated structures

**Next**: [Gazebo World Setup](../module-2-digital-twin/setup-gazebo.md) — Create environments for your humanoid to interact with.
