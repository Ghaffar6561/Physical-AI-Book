# Sensor Fusion & Localization

Humanoid robots live in complex, dynamic environments. A single sensor cannot see everything. Cameras see colors and shapes but fail in darkness. LiDAR sees depth but ignores texture. IMU detects motion but drifts over time. The solution: **sensor fusion**—combining multiple sensors to get a complete, accurate picture of the world.

This section teaches you how real robots build robust perception by fusing complementary sensor streams.

---

## Why Sensor Fusion Matters

### The Problem: Single Sensors Are Limited

**Vision (RGB Camera):**
- ✓ Rich information: color, texture, appearance
- ✗ Fails in darkness, bright sunlight, reflections
- ✗ No depth information without stereo
- ✗ Slow (30 Hz) with high latency (33 ms)

**LiDAR (3D Laser Scanning):**
- ✓ Fast depth measurement at 10-40 Hz
- ✓ Works in darkness, bright light (agnostic to lighting)
- ✗ No color information; hard to distinguish similar-looking objects
- ✗ Expensive ($5K-50K+)

**IMU (Inertial Measurement Unit):**
- ✓ Fast: 100-200 Hz
- ✓ Detects motion instantly
- ✗ Drifts over time (gravity bias accumulates)
- ✗ No absolute position information

**Wheel Odometry (Encoder-based Position):**
- ✓ Direct position feedback
- ✓ Fast and cheap
- ✗ Drifts on slippery surfaces
- ✗ Accumulates error over long distances

### Real Example: Humanoid Walking

```
Task: Walk 10 meters in a dark warehouse, avoid obstacles

Option 1: Vision only
  Problem: Can't see in darkness → robot bumps into wall ✗

Option 2: LiDAR only
  Problem: Detects wall but robot can't distinguish between wall and floor clutter
  Also: Reflections from wet floor confuse sensor ✗

Option 3: Fused perception (LiDAR + Vision + IMU)
  - LiDAR: Detects obstacles (wall, boxes)
  - Vision: Provides color info to distinguish materials (red wall vs reflection)
  - IMU: Maintains walking balance during motion
  Result: Robot navigates safely ✓
```

**Key insight**: Each sensor sees a different aspect of reality. Fusion combines these aspects into a coherent model.

---

## The Kalman Filter: Core Algorithm

Most modern robots use the **Kalman Filter** for sensor fusion. It's elegant and mathematically proven.

### The Idea (Conceptual)

The Kalman filter iterates two steps:

**1. Prediction Phase**
```
Model: "Based on last position and commanded velocity, where should the robot be now?"

Example:
  Last position: x = 5 m
  Commanded velocity: v = 1 m/s
  Time elapsed: Δt = 0.1 s
  Predicted position: x_pred = 5 + (1 × 0.1) = 5.1 m
  Uncertainty: ±0.05 m (model is roughly accurate)
```

**2. Correction Phase**
```
Sensor measurement: "What does the sensor actually measure?"

Example:
  LiDAR reports: x_meas = 5.15 m
  Sensor uncertainty: ±0.01 m (LiDAR is very accurate)

  Weighted blend:
    Trust prediction 95%, sensor 5%? No.
    Trust sensor more (lower uncertainty) → weight = 0.01 / (0.05 + 0.01) = 16.7%

  Updated estimate: x_fused = 0.833 × 5.1 + 0.167 × 5.15 = 5.108 m
```

The filter **automatically weights** sensors by their uncertainty. Accurate sensors get more weight.

### The Math (Intuitive)

**State equation (prediction):**
```
x(t) = A × x(t-1) + B × u(t) + w(t)

Where:
  x = robot state (position, velocity, etc.)
  A = state transition matrix (how motion happens)
  B = control input matrix (effect of commanded velocity)
  u = control input (velocity command)
  w = process noise (prediction uncertainty)
```

**Measurement equation (correction):**
```
z(t) = H × x(t) + v(t)

Where:
  z = sensor measurement
  H = measurement matrix (maps state to sensor space)
  v = measurement noise (sensor uncertainty)
```

**Kalman gain (how much to trust sensor):**
```
K(t) = P_pred(t) / (P_pred(t) + R)

Where:
  P_pred = prediction covariance (how uncertain we are about prediction)
  R = measurement covariance (how uncertain sensor is)

If R is small (sensor is very accurate), K is large (trust sensor more)
If P_pred is small (prediction is confident), K is small (trust prediction more)
```

**Update step:**
```
x_fused = x_pred + K × (z - H × x_pred)
         ^^^^^^^^     ^   ^^^^^^^^^^^^^^^^^^
         fused state  gain innovation (difference between measurement and prediction)
```

### Why Kalman Filter?

- **Optimal** for linear systems (minimizes error variance)
- **Fast** (runs in <1 ms even for high-dimensional state)
- **Principled** (based on probability theory, not heuristics)
- **Extensible** (Extended KF for nonlinear, Particle filters for non-Gaussian, etc.)

---

## Example: Humanoid Walking with IMU + Odometry Fusion

Let's build a practical example: fusing wheel odometry (position) with IMU (acceleration/rotation) to estimate humanoid pose during walking.

### Setup

```
Humanoid walking forward at 1 m/s

Sensors:
  1. Wheel encoders: Give position Δx every 10 ms (±5 mm uncertainty)
  2. IMU: Gives acceleration and angular velocity at 100 Hz (high noise, but fast)
  3. SLAM camera: Gives absolute position every 2 seconds (±10 cm uncertainty, but slow)

Goal: Estimate position with low latency AND high accuracy
```

### Code: ROS 2 Sensor Fusion Node

```python
"""
Sensor fusion for humanoid localization.

Fuses:
1. Wheel odometry (fast, drifts)
2. IMU (very fast, but noisy)
3. SLAM camera (slow, but accurate global reference)

Uses Extended Kalman Filter (EKF) to combine all three.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import math


class SensorFusionNode(Node):
    """
    Extended Kalman Filter for humanoid pose estimation.

    State: [x, y, theta, vx, vy, omega]
      x, y, theta = position and orientation
      vx, vy, omega = velocity and angular velocity
    """

    def __init__(self):
        super().__init__('sensor_fusion_ekf')

        # State estimate: [x, y, theta, vx, vy, omega]
        self.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Initial state

        # State covariance (uncertainty)
        self.P = np.eye(6) * 0.1  # Initially high uncertainty

        # Process noise (how much we expect the model to be wrong)
        self.Q = np.eye(6) * 0.01

        # Measurement noise covariance
        self.R_odom = np.eye(3) * 0.005  # Odometry: ±5 mm
        self.R_imu = np.eye(3) * 0.1    # IMU: noisy but fast
        self.R_slam = np.eye(3) * 0.01  # SLAM: accurate but slow

        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odometry_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.slam_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/slam/pose', self.slam_callback, 10
        )

        # Publisher for fused pose
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/pose/estimate', 10
        )

        self.last_time = self.get_clock().now()
        self.get_logger().info('Sensor Fusion EKF initialized')

    def odometry_callback(self, msg):
        """Update state estimate using wheel odometry (slow drift, low noise)."""
        t = self.get_clock().now()
        dt = (t - self.last_time).nanoseconds / 1e9

        # Extract odometry measurement: [x, y, theta]
        z = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            self._yaw_from_quaternion(msg.pose.pose.orientation)
        ])

        # Prediction step
        self.x, self.P = self._predict(dt)

        # Update step with odometry (low noise, trust it)
        self.x, self.P = self._update(z, self.R_odom, H=np.eye(3, 6))

        self.last_time = t
        self._publish_estimate()

    def imu_callback(self, msg):
        """Update using IMU (high frequency, high noise)."""
        # IMU gives acceleration and angular velocity
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        wz = msg.angular_velocity.z

        # Update velocity estimate (integrate acceleration)
        dt = 0.01  # IMU runs at 100 Hz
        self.x[3] += ax * dt  # vx
        self.x[4] += ay * dt  # vy
        self.x[5] = wz        # omega (angular velocity)

        # Increase uncertainty (IMU is noisy)
        self.P += np.eye(6) * 0.001

    def slam_callback(self, msg):
        """Correct drift using SLAM (slow but accurate global reference)."""
        # SLAM provides absolute position and orientation
        z = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            self._yaw_from_quaternion(msg.pose.pose.orientation)
        ])

        # Extract SLAM covariance
        R_slam = np.diag([0.01, 0.01, 0.01])  # ±10 cm

        # Update state with SLAM (very reliable)
        self.x, self.P = self._update(z, R_slam, H=np.eye(3, 6))

        self.get_logger().info(f'SLAM correction: x={self.x[0]:.2f}, y={self.x[1]:.2f}')
        self._publish_estimate()

    def _predict(self, dt):
        """Prediction step: integrate motion model."""
        x = self.x.copy()
        theta = x[2]
        vx = x[3]
        vy = x[4]

        # Kinematic model (differential drive)
        x[0] += vx * math.cos(theta) * dt
        x[1] += vy * math.sin(theta) * dt
        x[2] += x[5] * dt  # theta += omega * dt

        # Jacobian of motion model (for covariance propagation)
        F = np.eye(6)
        F[0, 2] = -vy * math.sin(theta) * dt
        F[0, 3] = math.cos(theta) * dt
        F[1, 2] = vy * math.cos(theta) * dt
        F[1, 4] = math.sin(theta) * dt
        F[2, 5] = dt

        # Propagate covariance: P = F P F^T + Q
        P = F @ self.P @ F.T + self.Q

        return x, P

    def _update(self, z, R, H):
        """Update step: correct prediction with measurement."""
        # Innovation (difference between measurement and prediction)
        y = z - H @ self.x

        # Innovation covariance: S = H P H^T + R
        S = H @ self.P @ H.T + R

        # Kalman gain: K = P H^T S^-1
        K = self.P @ H.T @ np.linalg.inv(S)

        # Updated state: x = x + K y
        x = self.x + K @ y

        # Updated covariance: P = (I - K H) P
        P = (np.eye(6) - K @ H) @ self.P

        return x, P

    def _publish_estimate(self):
        """Publish fused pose estimate with covariance."""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.pose.pose.position.x = self.x[0]
        msg.pose.pose.position.y = self.x[1]

        # Convert theta to quaternion
        q = self._quaternion_from_yaw(self.x[2])
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        # Covariance (from filter state)
        msg.pose.covariance = self.P.flatten().tolist()[:36]

        self.pose_pub.publish(msg)

    @staticmethod
    def _yaw_from_quaternion(q):
        """Extract yaw angle from quaternion."""
        return math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y**2 + q.z**2)
        )

    @staticmethod
    def _quaternion_from_yaw(yaw):
        """Convert yaw angle to quaternion."""
        return [
            0,  # x
            0,  # y
            math.sin(yaw / 2),  # z
            math.cos(yaw / 2)   # w
        ]


def main():
    rclpy.init()
    node = SensorFusionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### How It Works

1. **Odometry callback (10 Hz)**: Wheel encoders give position. Use these as measurements to correct drift.
   - Odometry drifts slowly, so high trust in measurements
   - After 10 minutes: ~10 m position error (acceptable)

2. **IMU callback (100 Hz)**: Acceleration and angular velocity arrive fast.
   - High frequency → can correct for sudden jerks
   - High noise → only use for short-term correction
   - Doesn't accumulate in state; just updates velocity

3. **SLAM callback (0.5 Hz)**: Camera-based localization provides global reference.
   - Very accurate (±10 cm)
   - Very slow (needs to process images)
   - Use to zero out long-term drift from odometry

**Result**: Position estimate is:
- **Fast** (IMU runs at 100 Hz)
- **Accurate** (SLAM corrects drift)
- **Robust** (works even if one sensor fails temporarily)

---

## Complementary Sensors Analysis

### Vision (RGB/Stereo Camera)

**Strengths:**
- Rich semantic information (object recognition)
- Texture and appearance matching
- Stereo depth without moving parts

**Weaknesses:**
- Slow (30 Hz)
- High latency (30-50 ms due to image processing)
- Fails in low light, bright sunlight, reflections
- Requires powerful compute (GPU)

**Fuse with:**
- LiDAR: LiDAR gives depth, vision gives semantic meaning
- Odometry: Vision drift corrected by odometry periodically
- IMU: IMU stabilizes short-term motion for vision tracking

**Example**: **Visual-Inertial Odometry (VIO)**
```
Fuses: Stereo camera (depth) + IMU (6-axis motion)
Result: Fast, accurate position (200 Hz) without GPS
Used in: AR/VR headsets, drones, phones
```

### LiDAR (3D Laser Scanner)

**Strengths:**
- Accurate depth (±2-5 cm)
- Works in darkness and bright light
- Fast (10-40 Hz)
- Robust to reflections

**Weaknesses:**
- No color/texture information
- Expensive ($5K-50K)
- Struggles with transparent surfaces (glass, water)
- Bulky (especially mechanical spinning types)

**Fuse with:**
- Vision: Adds color to distinguish similar-looking objects
- Odometry: Long-range point cloud matching corrects drift
- IMU: Motion compensation for scanning artifacts

**Example**: **LiDAR + Vision SLAM**
```
Fuses: LiDAR 3D point clouds + RGB images
Result: Semantic 3D map (not just points)
Example: "I see a red chair" not just "points at z=1.2m"
```

### Inertial Measurement Unit (IMU)

**Strengths:**
- Very fast (100-200 Hz)
- No external dependencies (GPS not needed)
- Cheap ($10-50)
- Detects motion instantly

**Weaknesses:**
- Gyro bias causes orientation drift (~1° per minute)
- Accel bias causes velocity drift (~0.1 m/s per minute)
- No absolute position (only relative motion)
- Noise is high

**Fuse with:**
- Magnetometer: Corrects gyro drift using Earth's magnetic field
- Barometer: Altitude for flying/climbing robots
- Odometry: Odometry constrains IMU drift
- Vision: Camera fixes orientation when available

**Example**: **Inertial-Measurement Unit + Magnetometer**
```
Fuses: Accel + Gyro (IMU) + Magnetometer (compass)
Result: Stable orientation estimate
Drift: Reduced from 1°/min to 0.1°/min
```

### Wheel Encoders (Odometry)

**Strengths:**
- Direct position feedback
- Very accurate over short distance (±1 cm per meter)
- Cheap and reliable
- Low power

**Weaknesses:**
- Drifts on slippery surfaces (wheels slip)
- Accumulates error over distance
- Only works on ground robots
- Assumes wheels don't slip

**Fuse with:**
- IMU: IMU detects skidding (acceleration != velocity derivative)
- Vision: Periodic vision updates reset drift
- LiDAR: Loop-closure detection (e.g., "we're back at starting point")

**Example**: **Encoder + IMU Fusion**
```
Normal walking: Encoders predict position, IMU validates motion
Slippery surface: Encoder estimate drifts, IMU detects skid (accel ≠ velocity)
Controller: "Wheels slipping, increase traction"
```

---

## SLAM: Simultaneous Localization and Mapping

**SLAM** is the problem of building a map while simultaneously knowing where you are on that map. All modern mobile robots solve this.

### The SLAM Loop

```
1. Sense: Acquire sensor data (camera, LiDAR, etc.)
   ↓
2. Process: Extract features (corners, edges, planes)
   ↓
3. Match: Find correspondence between current and previous observations
   ("This corner I see now matches the corner I saw 5 meters ago")
   ↓
4. Optimize: Update robot pose and map jointly to satisfy all observations
   ↓
5. Loop closure: Detect when returning to a previously visited location
   ("I recognize this hallway, it's where I started!")
   ↓
6. Refine: Reoptimize entire map using loop-closure constraint
```

### Practical SLAM Algorithms for Humanoid Robots

**ORB-SLAM3** (Visual SLAM)
- Input: RGB camera + optional IMU
- Output: Camera pose + 3D point cloud map
- Robustness: Good
- Compute: Moderate (CPU)
- Accuracy: ±1-5% of distance
- Used by: Boston Dynamics, Tesla (humanoid projects)

**LOAM** (LiDAR SLAM)
- Input: LiDAR point cloud
- Output: Robot pose + dense 3D map
- Robustness: Excellent (works in any lighting)
- Compute: Low (CPU)
- Accuracy: ±1-2% of distance
- Used by: Autonomous vehicles, ground robots

**Cartographer** (Graph-based SLAM)
- Input: LiDAR + IMU + optional odometry
- Output: 2D/3D maps with loop closures
- Robustness: Excellent
- Compute: Moderate
- Accuracy: ±0.5-1% of distance
- Used by: Mobile robots, Spot (Boston Dynamics)

### Running SLAM on a Humanoid

```bash
# Launch navigation with Cartographer SLAM
ros2 launch humanoid_sim cartographer.launch.py

# Monitor map quality
rviz2 -d humanoid_sim/rviz_slam.rviz

# When done, save map
ros2 run nav2_map_server map_saver_cli -f ~/humanoid_map
```

---

## Integration Example: Real Humanoid Perception Stack

```
Camera → Vision processing (ORB-SLAM)
         ↓
         ├→ 3D pose estimate (3D points matched)
         ├→ Confidence (# of matched features)
         ↓
LiDAR  → Plane/edge detection (Cartographer)
         ↓
         ├→ Surface normals
         ├→ Obstacle detection
         ↓
IMU    → Orientation (gyro + accelerometer)
         ↓
Odometry → Position (from motor commands)
           ↓
           ├→ Velocity feedback
           ├→ Slip detection
           ↓
Sensor Fusion (Kalman Filter)
           ↓
           ├→ Fused pose (x, y, z, roll, pitch, yaw)
           ├→ Velocity (vx, vy, vz, ωx, ωy, ωz)
           ├→ Confidence (covariance)
           ↓
Navigation/Control Stack
           ↓
           → Motion planning
           → Collision avoidance
           → Manipulation
```

This is how real robots perceive their world: not with one sensor, but with all of them working together.

---

## Key Takeaways

| Concept | Definition | When to Use |
|---------|-----------|-------------|
| **Sensor Fusion** | Combining multiple sensors to get better estimates | Always (single sensors insufficient) |
| **Kalman Filter** | Optimal algorithm for fusing linear measurements with noise | Position/velocity/orientation estimation |
| **SLAM** | Building a map while estimating robot pose | Navigation in unknown environments |
| **Complementary Sensors** | Sensors that fail in different conditions (vision + LiDAR) | Robust perception in varied environments |
| **Sensor Uncertainty** | Covariance matrices quantifying measurement noise | Automatic sensor weighting in fusion |
| **Loop Closure** | Detecting return to previously visited location | Correcting long-term drift |

---

## Further Reading

- **"Multi-Sensor Fusion for Joint Tracking"** — Welch & Bishop (probabilistic theory)
- **"ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM"** — Campos et al. (state-of-the-art visual SLAM)
- **"Cartographer: A System for Real-Time SLAM"** — Hess et al. (Google Cartographer paper)
- **"The Problem of Mobile Sensing"** — Thrun (foundational SLAM reference)

---

**Next**: [Sim-to-Real Transfer Strategies](sim-to-real-transfer.md) — Now that you understand sensor fusion, learn how to train perception models in simulation and deploy them on real robots.

You now understand how real robots combine multiple noisy sensors into confident, accurate perception. This is the foundation for deploying sim-trained policies to real hardware.
