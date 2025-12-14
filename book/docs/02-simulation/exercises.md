# Module 2: Exercises & Solutions

These exercises guide you through using Gazebo to simulate a humanoid robot. You'll load models, inspect sensor output, modify URDFs, and understand sim-to-real transfer challenges.

**Instructions**: Try each exercise first, then compare with the solution. Don't look at solutions until you've attempted the exercise!

---

## Exercise 1: Load and Run the Humanoid in Gazebo

### Objective

Successfully launch the humanoid robot in Gazebo and verify that:
1. The robot loads without errors
2. Physics simulation is stable
3. All sensors are publishing data

### Task

1. **Start Gazebo with the provided world**:
   ```bash
   cd ~/PhysicalAI-Book/book/examples/humanoid-sim
   gazebo --verbose -s libgazebo_ros_init.so gazebo_models/simple_world.sdf
   ```

2. **Spawn the humanoid robot** (in another terminal):
   ```bash
   ros2 service call /spawn_entity gazebo_msgs/SpawnEntity \
     "{name: 'humanoid', xml: '$(cat gazebo_models/humanoid_simple.urdf)'}"
   ```

3. **Launch the control nodes**:
   ```bash
   python3 book/static/code-examples/ros2_humanoid_nodes.py
   ```

4. **Verify sensor data** (in another terminal):
   ```bash
   # List all topics
   ros2 topic list

   # Check joint states are updating
   ros2 topic hz /joint_states

   # Check joint commands are being sent
   ros2 topic echo /joint_commands --once
   ```

### Success Criteria

- ✅ Gazebo window shows humanoid standing upright
- ✅ Robot doesn't collapse or fall through floor
- ✅ `/joint_commands` topic shows 15+ joints
- ✅ `/joint_states` publishing at 100+ Hz
- ✅ No physics errors in Gazebo output
- ✅ Control nodes running without crashes

### Hints

- If the robot falls through the floor: check the `<inertia>` values in the URDF (they might be too small)
- If Gazebo runs slowly: reduce sensor update rates in the world file
- If topics aren't publishing: verify ROS 2 bridge plugins are loaded (`libgazebo_ros_*.so`)

### Solution

The exercise succeeds when:

```bash
$ ros2 topic hz /joint_states
average rate: 100.05 Hz
$ ros2 topic list | grep -E "(camera|joint|gripper)"
/camera/image_raw
/gripper_commands
/joint_command_vision
/joint_commands
/joint_states
```

The humanoid should be in Gazebo standing upright with arms moving in smooth sinusoidal patterns.

**Common Issues & Fixes:**

| Issue | Cause | Fix |
|-------|-------|-----|
| Robot collapses immediately | Gravity is too strong or inertia too small | Check inertia values: Ixx, Iyy, Izz should be > 0.01 for torso |
| Gazebo runs at 10% speed | Too many objects or high update rates | Reduce camera/LiDAR update rate to 10 Hz, use GPU LiDAR |
| Plugins not loading | ROS 2 bridge not installed | `sudo apt install ros-humble-gazebo-ros` |
| Joint commands ignored | Joint names don't match URDF | Check `joint_names` list matches URDF exactly |

---

## Exercise 2: Modify the URDF — Add a Sensor

### Objective

Understand how URDF describes robot structure by adding a new sensor to the humanoid.

### Task

You are tasked to add an **IMU (Inertial Measurement Unit)** to the humanoid's torso. The IMU should:
- Be attached to the torso (fixed joint)
- Publish 6-axis data: 3-axis acceleration + 3-axis angular velocity
- Have realistic noise parameters

**Steps:**

1. Open `book/examples/humanoid-sim/gazebo_models/humanoid_simple.urdf`

2. Find the `</robot>` closing tag (near the end)

3. Before `</robot>`, add an IMU link and joint:
   ```xml
   <!-- IMU attached to torso center -->
   <joint name="imu_joint" type="fixed">
     <parent link="torso"/>
     <child link="imu_link"/>
     <origin xyz="0 0 0" rpy="0 0 0"/>
   </joint>

   <link name="imu_link">
     <inertial>
       <mass value="0.05"/>  <!-- Very light -->
       <inertia ixx="0.0001" iyy="0.0001" izz="0.0001"
                ixy="0" ixz="0" iyz="0"/>
     </inertial>
     <visual>
       <geometry>
         <box size="0.03 0.03 0.03"/>  <!-- Tiny sensor -->
       </geometry>
       <material name="sensor_material">
         <color rgba="0.2 0.2 0.2 1.0"/>
       </material>
     </visual>
     <collision>
       <geometry>
         <box size="0.03 0.03 0.03"/>
       </geometry>
     </collision>
   </link>
   ```

4. **Validate the URDF**:
   ```bash
   check_urdf humanoid_simple.urdf
   ```
   Should output: `robot name is: humanoid_simple` with no errors

5. **Test it in Gazebo**:
   ```bash
   # Spawn the modified robot
   ros2 service call /spawn_entity gazebo_msgs/SpawnEntity \
     "{name: 'humanoid_imu', xml: '$(cat humanoid_simple.urdf)'}"
   ```

### Success Criteria

- ✅ URDF is valid (check_urdf passes)
- ✅ Small black cube appears on the robot's torso in Gazebo
- ✅ IMU link appears in `/tf` (transform tree)
- ✅ Joint limits parse correctly

### Hints

- A "fixed" joint means the IMU doesn't move relative to the torso
- Mass should be very small (0.05 kg = 50 grams) for a real IMU
- Inertia values should also be tiny for a light sensor
- The origin xyz="0 0 0" means the IMU is at the center of the torso

### Solution

**Added URDF snippet** (at the end, before `</robot>`):

```xml
<!-- Inertial Measurement Unit (IMU) attached to torso -->
<joint name="imu_joint" type="fixed">
  <parent link="torso"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>  <!-- At torso center -->
</joint>

<link name="imu_link">
  <inertial>
    <mass value="0.05"/>  <!-- 50 grams - realistic IMU weight -->
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.0001" iyy="0.0001" izz="0.0001"
             ixy="0" ixz="0" iyz="0"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.03 0.03 0.03"/>  <!-- 3 cm cube, realistic IMU size -->
    </geometry>
    <material name="imu_material">
      <color rgba="0.2 0.2 0.2 1.0"/>  <!-- Dark gray/black -->
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.03 0.03 0.03"/>
    </geometry>
  </collision>
</link>
```

**Verification**:

```bash
$ check_urdf humanoid_simple.urdf
robot name is: humanoid_simple
---------- Successfully Parsed XML ---------------
root Link: torso
    child(0):  head
        child(0):  camera_link
    child(1):  right_arm_upper
        child(0):  right_arm_lower
            child(0):  right_wrist
                child(0):  right_hand
    child(2):  left_arm_upper
        ...
    child(N):  imu_link
---------- Link Tree Has 25 Branches ----------
```

**Key Insight**: Adding a sensor is just adding a `<link>` connected by a `<joint>`. The physics engine treats it like any other link, but with tiny mass/inertia since it's passive.

---

## Exercise 3: Analyze Sim-to-Real Transfer Challenges

### Objective

Identify and document the differences between your simulation and a real humanoid robot. Understand what could break during sim-to-real transfer.

### Task

For each category below, describe:
1. **What's different** between simulation and reality
2. **Why it matters** (impact on robot behavior)
3. **How to mitigate** (domain randomization, robust control, etc.)

**Categories to analyze:**

#### A. Physics Inaccuracies

**Simulation version:**
```python
# Gazebo uses Coulomb friction model
friction = mu * normal_force
```

**Real world version:**
```python
# Real friction depends on:
# - Surface roughness (texture, dust)
# - Temperature (affects lubricant viscosity)
# - Velocity (static vs kinetic friction)
# - Material properties (rubber vs metal)
```

**Your analysis** (fill in 2-3 sentences):
- What's different? ...
- Why it matters? ...
- Mitigation? ...

#### B. Sensor Noise and Lag

**Simulation camera:**
- Perfect 30 Hz updates
- Ideal color accuracy
- No motion blur

**Real camera:**
- Gaussian + quantization noise
- Rolling shutter artifacts
- Actual 33 ms latency (some frames miss)

**Your analysis**:
- What's different? ...
- Why it matters? ...
- Mitigation? ...

#### C. Actuator Dynamics

**Simulation joints:**
```python
# Position commands executed instantly
desired_position = 0.5 rad
actual_position = 0.5 rad  # Immediate!
```

**Real motors:**
```python
# Motor accelerates over time
# Takes 10-50 ms to reach desired velocity
# Current/voltage limits constrain acceleration
# Backlash and friction affect precision
```

**Your analysis**:
- What's different? ...
- Why it matters? ...
- Mitigation? ...

### Success Criteria

- ✅ You've identified at least 3 major sim-to-real gaps
- ✅ You understand why each matters (not just "it's different")
- ✅ You've proposed a mitigation strategy for each

### Solution

**Example Analysis** (use as reference):

#### A. Physics Inaccuracies

**What's different**:
- Gazebo uses a single friction coefficient μ. Real surfaces have velocity-dependent friction, temperature effects, and roughness variation.
- Gazebo contact solver approximates; real collisions have complex stress distributions.

**Why it matters**:
- A humanoid grasping an object with calculated friction force (e.g., 50 N) might slip on real surfaces with different friction (e.g., wet, dusty, or very smooth).
- Walking gaits optimized in sim may have unstable contact in reality.

**Mitigation**:
- **Domain randomization**: Train with varied friction coefficients (μ = 0.3 to 0.8)
- **Robust control**: Add safety margins to force/torque commands
- **Real data**: Measure friction on test surface, update URDF
- **Perceptual feedback**: Use force sensors to detect slip, adjust grip dynamically

#### B. Sensor Noise and Lag

**What's different**:
- Gazebo cameras have perfect timing and can optionally add Gaussian noise. Real cameras have 33 ms latency, rolling shutter (different rows at different times), motion blur, and hot pixels.
- Gazebo can render perfect images; real cameras have lens distortion, chromatic aberration, and lighting variations.

**Why it matters**:
- Vision-based grasping might fail on real robot if CNN trained on perfect Gazebo images encounters rolling shutter artifacts
- Motion blur from fast arm movement causes false detections
- Latency causes the robot to be "blind" during rapid motions (perception lag > control latency)

**Mitigation**:
- **Sensor simulation**: Add realistic camera plugins (rolling shutter, lens distortion)
- **Training augmentation**: Train perception on varied lighting, blur, and noise
- **Control design**: Use wider safety margins, slower speeds if latency is significant
- **Real validation**: Test perception on real camera images captured in sim-like conditions

#### C. Actuator Dynamics

**What's different**:
- Gazebo allows position setpoints to be executed instantly. Real servos accelerate: τ = I × dω/dt. Joint takes time to reach desired position.
- Real motors have max torque/velocity limits that must be respected.

**Why it matters**:
- A trajectory that works in sim (commanding 10 rad/s instantly) will fail on real hardware (motor can only do 2 rad/s max)
- Aggressive control works in sim but causes instability on real robot due to lag
- Backlash and friction cause position tracking error on real joints

**Mitigation**:
- **Actuator model**: Add motor inertia and acceleration limits to simulation
- **Realistic saturation**: Cap joint velocities and torques in controller
- **Trajectory filtering**: Smooth trajectories with ramping (not step commands)
- **Tuning on hardware**: PID gains must be retuned on real robot (sim ≠ reality)

---

## Challenge Exercises (Optional)

### Challenge 1: Domain Randomization

Implement domain randomization by modifying the Gazebo world file to:
- Randomize object positions on the table
- Randomize friction coefficients
- Randomize lighting direction and intensity
- Randomize object colors

**Goal**: Train a grasping policy in sim with randomization, then test on real robot without retraining.

### Challenge 2: Sim-to-Real Validation Checklist

Create a checklist for validating a humanoid policy before deploying to real hardware:

```
[ ] Perception works with real camera images (not just simulated)
[ ] Trajectories are smooth (no jerky commands)
[ ] Safety margins are conservative (never command at limits)
[ ] Force/torque feedback is checked (no overload conditions)
[ ] Sim inertias match CAD specs (measure with real robot if possible)
[ ] Friction coefficients are realistic (test on real surfaces)
[ ] Latency budget is respected (perception + control < 100 ms)
```

### Challenge 3: System Identification

Use the humanoid in simulation to:
1. Apply known force/torque inputs
2. Measure resulting motions
3. Estimate physical parameters (mass, inertia, friction)
4. Compare to ground truth URDF values

This mimics real-world system identification where you'd use sensors to validate a model.

---

## Further Reading

### On Simulation and Robotics
- Brockman et al.: "OpenAI Gym" — Standardized sim environments
- Tobin et al.: "Domain Randomization for Transferring Deep Neural Networks from Simulation to Real World"
- Plappert et al.: "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"

### On Gazebo
- [Gazebo Official Documentation](http://gazebosim.org/)
- [ROS 2 Gazebo Integration](https://github.com/ros-simulation/gazebo_ros_pkgs)
- Gazebo Tutorials on URDFs, plugins, sensors

### On Digital Twins
- Tao et al.: "Digital Twins for Manufacturing" (Industry 4.0 perspective)
- Mirror World Computing: Digital twins for robotics research

---

## Summary

You've now learned:

✅ How to load and run a humanoid in Gazebo
✅ How to modify robot descriptions (URDF)
✅ How to identify sim-to-real gaps and mitigate them
✅ The limitations of simulation and how to work with them

**Key Insight**: Simulation is powerful because it's fast and safe, but it's *always* different from reality. The goal is not perfect fidelity, but sufficient fidelity for your control algorithm to transfer.

---

**Ready for the next module?** → [Module 3: Perception & Sim-to-Real Transfer](../03-perception/intro.md)

You've completed the foundations of humanoid robotics simulation. Next, we'll learn how to perceive the environment and transfer what we learned in simulation to real robots.
