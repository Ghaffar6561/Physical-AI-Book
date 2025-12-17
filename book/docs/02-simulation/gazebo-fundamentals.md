# Gazebo Fundamentals

Gazebo is the industry-standard open-source 3D robotics simulator. It combines:

- **Physics simulation** — Accurate rigid body dynamics with ODE or Bullet
- **Sensor simulation** — Cameras, LiDAR, IMU with realistic noise and lag
- **ROS 2 integration** — Native bridges for topics, services, and actions
- **Visualization** — 3D GUI and command-line tools
- **Scriptability** — Plugins for custom behavior and automation

This section teaches the core concepts you need to simulate humanoid robots and connect them to ROS 2 control systems.

---

## Core Concepts

### The Physics Engine

Gazebo's physics engine simulates rigid body dynamics at a fixed timestep:

```
At each simulation step (dt = 0.001 seconds = 1 kHz):

1. Apply forces/torques
   ├─ Gravity (9.81 m/s² downward)
   ├─ Joint motor commands (from ROS 2)
   └─ Contact forces (from collisions)

2. Compute accelerations
   └─ Using F = ma (Newton's second law)
   └─ Including inertia from URDF

3. Integrate motion
   ├─ New velocity = old velocity + acceleration × dt
   └─ New position = old position + velocity × dt

4. Detect collisions
   ├─ Check all object pairs
   ├─ Compute contact points and normals
   └─ Apply friction and restitution

5. Publish results to ROS 2
   └─ Joint states, sensor data, contact information
```

**Important**: The simulation timestep (1 ms) is much faster than real-time with modern hardware. One simulation second takes ~10-100 milliseconds of wall-clock time depending on complexity.

### World Files (SDF Format)

A world file (`.sdf`) describes the entire simulation environment:

**Structure of an SDF World:**

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">

    <!-- Physics engine settings -->
    <physics type="ode">
      <gravity>0 0 -9.81</gravity>
      <max_step_size>0.001</max_step_size>  <!-- 1 kHz simulation -->
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Ground plane (static object) -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>0.5 0.5 -1</direction>
    </light>

    <!-- Example: A simple ball object -->
    <model name="ball">
      <pose>0 0 0.5 0 0 0</pose>  <!-- x y z roll pitch yaw -->
      <link name="ball_link">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.01</ixx>
            <iyy>0.01</iyy>
            <izz>0.01</izz>
          </inertia>
        </inertial>
        <collision name="ball_collision">
          <geometry>
            <sphere><radius>0.1</radius></sphere>
          </geometry>
        </collision>
        <visual name="ball_visual">
          <geometry>
            <sphere><radius>0.1</radius></sphere>
          </geometry>
          <material>
            <diffuse>1 0 0 1</diffuse>  <!-- Red -->
          </material>
        </visual>
      </link>
      <!-- Add friction -->
      <surface>
        <friction>
          <ode>
            <mu>0.5</mu>      <!-- Coefficient of friction -->
            <mu2>0.5</mu2>
          </ode>
        </friction>
      </surface>
    </model>

    <!-- Example: A robot (loaded from URDF) -->
    <include>
      <uri>model://humanoid_simple</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>

  </world>
</sdf>
```

**Key SDF Elements:**

| Element | Purpose | Example |
|---------|---------|---------|
| `<physics>` | Engine settings, gravity, timestep | `<max_step_size>0.001</max_step_size>` |
| `<world>` | Container for all objects | `name="default"` |
| `<model>` | Robot or object | `name="humanoid_simple"` |
| `<link>` | Rigid body within a model | `name="torso"` |
| `<joint>` | Connection between links | `type="revolute"` (in URDF, converted to SDF) |
| `<inertial>` | Mass and inertia tensor | `<mass>10.0</mass>` |
| `<collision>` | Shape for collision detection | `<geometry><box><size>...</size></box>` |
| `<visual>` | Shape for rendering | Same geometry as collision |
| `<sensor>` | Camera, LiDAR, IMU | `<type>camera</type>` |
| `<plugin>` | C++ code executed in simulation | ROS 2 bridges, custom physics |

### Physics Engine Options

**ODE (Open Dynamics Engine)**
- Pros: Fast, stable, widely used
- Cons: Less accurate for high-speed collisions
- Use for: Real-time robot control, training
- Default in Gazebo

**Bullet**
- Pros: More accurate collisions, better contact dynamics
- Cons: Slower than ODE
- Use for: High-precision grasping, contact analysis
- Can be enabled: `<physics type="bullet">`

For humanoid manipulation, ODE is usually sufficient.

---

## Sensors in Gazebo

### Camera Sensor

A camera publishes RGB images to a ROS 2 topic.

**In SDF (attached to a link):**

```xml
<sensor name="camera" type="camera">
  <camera>
    <!-- Field of view and resolution -->
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>  <!-- RGB -->
    </image>
    <!-- Depth of field -->
    <clip>
      <near>0.01</near>    <!-- 1 cm minimum -->
      <far>100</far>       <!-- 100 m maximum -->
    </clip>
    <!-- Distortion (optional) -->
    <distortion>
      <k1>0.0</k1>
      <k2>0.0</k2>
      <k3>0.0</k3>
      <p1>0.0</p1>
      <p2>0.0</p2>
    </distortion>
  </camera>

  <!-- Noise model -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.01</stddev>  <!-- Standard deviation of pixel noise -->
  </noise>

  <!-- ROS 2 bridge plugin -->
  <plugin filename="libgazebo_ros_camera.so" name="camera_controller">
    <ros>
      <!-- Publish to this topic -->
      <remapping>~/image_raw:=camera/image_raw</remapping>
      <remapping>~/camera_info:=camera/camera_info</remapping>
    </ros>
    <camera_name>camera</camera_name>
    <frame_name>camera_link</frame_name>
    <update_rate>30</update_rate>  <!-- 30 Hz -->
  </plugin>
</sensor>
```

**ROS 2 Output:**
- Topic: `/camera/image_raw` (sensor_msgs/msg/Image)
- Topic: `/camera/camera_info` (sensor_msgs/msg/CameraInfo)
- Frequency: 30 Hz
- Message contains: Width, height, pixel data, encoding, intrinsic calibration

### LiDAR Sensor

A LiDAR (Light Detection and Ranging) measures distances to objects.

```xml
<sensor name="lidar" type="gpu_lidar">  <!-- gpu_lidar for performance -->
  <lidar>
    <scan>
      <horizontal>
        <samples>640</samples>  <!-- 640 rays per scan -->
        <resolution>1.0</resolution>
        <min_angle>-3.14</min_angle>  <!-- -180° -->
        <max_angle>3.14</max_angle>   <!-- +180° -->
      </horizontal>
      <vertical>
        <samples>16</samples>  <!-- 16 vertical layers -->
        <resolution>1.0</resolution>
        <min_angle>-0.26</min_angle>  <!-- -15° (looking slightly down) -->
        <max_angle>0.26</max_angle>   <!-- +15° (looking slightly up) -->
      </vertical>
    </scan>
    <range>
      <min>0.15</min>    <!-- 15 cm minimum range -->
      <max>50</max>      <!-- 50 m maximum range -->
      <resolution>0.01</resolution>  <!-- 1 cm resolution -->
    </range>
    <!-- Doppler effect (optional) -->
    <enable_metrics>false</enable_metrics>
  </lidar>

  <!-- Noise: typical LiDAR noise is gaussian in range -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.01</stddev>  <!-- 1 cm std dev -->
  </noise>

  <!-- ROS 2 bridge plugin -->
  <plugin filename="libgazebo_ros_gpu_lidar.so" name="lidar_controller">
    <ros>
      <remapping>~/scan:=scan</remapping>
      <remapping>~/points:=points</remapping>
    </ros>
    <frame_name>lidar_link</frame_name>
  </plugin>
</sensor>
```

**ROS 2 Output:**
- Topic: `/scan` (sensor_msgs/msg/LaserScan) — 2D projection
- Topic: `/points` (sensor_msgs/msg/PointCloud2) — Full 3D point cloud
- Frequency: Configurable (typically 10-20 Hz)

### IMU Sensor (Inertial Measurement Unit)

Measures linear acceleration and angular velocity.

```xml
<sensor name="imu" type="imu">
  <imu>
    <!-- Accelerometer specs -->
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>  <!-- ~0.06 °/s -->
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </z>
    </angular_velocity>

    <!-- Accelerometer specs -->
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>  <!-- ~0.1 m/s² -->
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>

  <!-- ROS 2 bridge plugin -->
  <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_controller">
    <ros>
      <remapping>~/imu:=imu</remapping>
      <remapping>~/imu/data:=imu/data</remapping>
    </ros>
    <frame_name>imu_link</frame_name>
  </plugin>
</sensor>
```

**ROS 2 Output:**
- Topic: `/imu` or `/imu/data` (sensor_msgs/msg/Imu)
- Frequency: 100-200 Hz (high-bandwidth)
- Message contains: Linear acceleration (3 axes), angular velocity (3 axes), covariance

---

## Running Gazebo with ROS 2

### Basic Gazebo Workflow

**Step 1: Create a launch file** (`launch/gazebo.launch.py`)

```python
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the world file
    world_file = PathJoinSubstitution(
        [
            get_package_share_directory('humanoid_sim'),
            'worlds',
            'simple_world.sdf'
        ]
    )

    # Start Gazebo server (physics simulation, no GUI)
    gazebo_server = ExecuteProcess(
        cmd=[
            'gazebo',
            '--verbose',
            '-s', 'libgazebo_ros_init.so',  # ROS 2 init plugin
            '-s', 'libgazebo_ros_factory.so',  # ROS 2 spawn plugin
            world_file
        ],
        output='screen'
    )

    # Start Gazebo client (GUI visualization)
    gazebo_client = ExecuteProcess(
        cmd=['gzclient'],
        output='screen'
    )

    # Include robot state publisher
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('humanoid_sim'),
            '/launch/state_publisher.launch.py'
        ])
    )

    return LaunchDescription([
        gazebo_server,
        gazebo_client,
        robot_state_publisher,
    ])
```

**Step 2: Run the launch file**

```bash
ros2 launch humanoid_sim gazebo.launch.py
```

**Step 3: Verify sensors are publishing**

```bash
# List all topics
ros2 topic list

# Watch camera frames
ros2 topic echo /camera/image_raw --once

# Watch LiDAR point cloud
ros2 topic echo /points --once

# Watch IMU
ros2 topic echo /imu --once
```

### Headless Simulation (No GUI)

For testing and CI/CD, run Gazebo without the GUI:

```bash
gazebo --verbose -s libgazebo_ros_init.so -s libgazebo_ros_factory.so \
  /path/to/world.sdf
```

Or in a launch file:

```python
gazebo_server = ExecuteProcess(
    cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so',
         '-s', 'libgazebo_ros_factory.so', world_file],
    output='screen'
)
# Skip gazebo_client (GUI)
```

---

## Controlling the Robot in Gazebo

### Publishing Joint Commands

A ROS 2 node can command joint positions or velocities:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Publisher for joint state commands
        self.joint_command_pub = self.create_publisher(
            JointState, '/joint_command', 10
        )

        # Timer to send commands at 50 Hz
        self.timer = self.create_timer(0.02, self.send_joint_command)
        self.time_step = 0

    def send_joint_command(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        # Joint names must match URDF
        msg.name = [
            'shoulder_pitch', 'shoulder_roll', 'elbow_pitch',
            'hip_pitch', 'knee_pitch', 'ankle_pitch'
        ]

        # Command positions (rad) — smooth sinusoid for testing
        t = self.time_step * 0.02  # seconds
        msg.position = [
            0.5 * math.sin(t),      # shoulder pitch
            0.3 * math.cos(t),      # shoulder roll
            0.2 * math.sin(2*t),    # elbow pitch
            0.5 * math.sin(t),      # hip pitch
            1.0 * math.sin(t),      # knee pitch
            0.3 * math.cos(t),      # ankle pitch
        ]

        # Effort (motor torque) — optional
        msg.effort = [10.0] * 6

        self.joint_command_pub.publish(msg)
        self.time_step += 1

def main():
    rclpy.init()
    controller = RobotController()
    rclpy.spin(controller)

if __name__ == '__main__':
    main()
```

### Receiving Joint State Feedback

The Gazebo bridge publishes joint states from the simulation:

```python
from sensor_msgs.msg import JointState

class RobotMonitor(Node):
    def __init__(self):
        super().__init__('robot_monitor')

        # Subscribe to actual joint states from Gazebo
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

    def joint_state_callback(self, msg):
        # msg.name: list of joint names
        # msg.position: current position (rad)
        # msg.velocity: current velocity (rad/s)
        # msg.effort: current torque (Nm)

        for i, name in enumerate(msg.name):
            pos = msg.position[i] if i < len(msg.position) else 0.0
            vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
            torque = msg.effort[i] if i < len(msg.effort) else 0.0

            self.get_logger().info(
                f'{name}: pos={pos:.3f} rad, vel={vel:.3f} rad/s, '
                f'torque={torque:.3f} Nm'
            )
```

---

## Common Gazebo Issues and Solutions

### Issue 1: Physics Simulation is Unstable

**Problem**: Robot falls through floor or exhibits unrealistic bouncing

**Causes:**
- Timestep too large (integration error)
- Inertia values wrong (too small → unstable)
- Collision geometry doesn't match visual geometry

**Solutions:**
```xml
<!-- Reduce timestep for better accuracy -->
<max_step_size>0.0005</max_step_size>  <!-- 0.5 ms instead of 1 ms -->

<!-- Verify inertia is reasonable (Ixx, Iyy, Izz > 0) -->
<!-- For a box of mass m, width w, depth d, height h: -->
<!-- Ixx = m * (d² + h²) / 12 -->
```

### Issue 2: Sensors Publish No Data

**Problem**: Topics exist but no messages arrive

**Causes:**
- Plugin not loaded (missing `<plugin>` tag in SDF)
- Update rate too high (simulation can't keep up)
- ROS 2 bridge not initialized

**Solutions:**
```bash
# Verify plugin is loaded
gazebo -s libgazebo_ros_init.so -s libgazebo_ros_factory.so world.sdf

# Check topics are published
ros2 topic list

# Check topic frequency
ros2 topic hz /camera/image_raw
```

### Issue 3: Slow Simulation Speed

**Problem**: Simulation runs in slow-motion (real_time_factor < 1.0)

**Causes:**
- Physics complexity (too many objects, contacts)
- High sensor update rates
- GPU not being used

**Solutions:**
```xml
<!-- Use simplified collision geometry -->
<collision>
  <geometry><box><!-- smaller/fewer boxes --></box></geometry>
</collision>

<!-- Lower sensor update rates -->
<update_rate>10</update_rate>  <!-- Instead of 30 Hz -->

<!-- Use GPU-accelerated LiDAR -->
<sensor type="gpu_lidar">  <!-- Not cpu_lidar -->
```

---

## Key Concepts Summary

| Concept | Purpose | Example |
|---------|---------|---------|
| **World (SDF)** | Describes entire simulation | `simple_world.sdf` with ground, lights, objects |
| **Model** | Robot or object in world | Humanoid, ball, table |
| **Link** | Rigid body segment | Torso, arm, leg |
| **Joint** | Connection between links | Revolute (rotate), prismatic (slide) |
| **Sensor** | Measures environment | Camera (images), LiDAR (depth), IMU (acceleration) |
| **Plugin** | ROS 2 bridge to Gazebo | Publishes sensor data, subscribes to commands |
| **Physics Engine** | Simulates forces/motion | ODE (fast) or Bullet (accurate) |
| **Timestep** | Simulation update interval | 0.001 s (1 ms, 1 kHz) |

---

## Your Gazebo Workflow

```
1. Create URDF robot model
2. Create SDF world file with ground, objects, physics
3. Create ROS 2 launch file to start Gazebo + bridge plugins
4. Write control nodes that:
   ├─ Subscribe to sensor topics (/camera, /lidar, /imu, /joint_states)
   ├─ Compute commands (perception → planning → control)
   └─ Publish commands (/joint_command, /gripper_command)
5. Monitor simulation with ROS 2 CLI and RViz
6. Validate physics plausibility (does behavior match expectations?)
7. Test sim-to-real transfer (can real robot use same code?)
```

---

**Next**: [URDF for Humanoids](urdf-humanoid.md) — Learn to model robot structure and kinematics.

Now you understand how Gazebo simulates physics and sensors. Let's learn to describe robots with URDF.
