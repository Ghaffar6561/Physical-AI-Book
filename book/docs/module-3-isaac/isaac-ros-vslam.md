---
sidebar_position: 3
title: "Isaac ROS VSLAM"
---

# Isaac ROS Visual SLAM (cuVSLAM)

## Overview

**Visual SLAM** (Simultaneous Localization and Mapping) allows a robot to build a map of its environment while simultaneously tracking its position within that map. NVIDIA's **cuVSLAM** provides GPU-accelerated visual odometry that runs in real-time on Jetson and desktop GPUs.

## Why VSLAM for Humanoids?

| Challenge | VSLAM Solution |
|-----------|----------------|
| **No GPS indoors** | Visual features provide localization |
| **Dynamic environments** | Continuous map updates |
| **Precise manipulation** | Centimeter-level positioning |
| **Human interaction** | Real-time response to environment changes |

## cuVSLAM Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    cuVSLAM Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐          │
│   │  Stereo  │───▶│ Feature  │───▶│    Visual    │          │
│   │  Camera  │    │ Extraction│    │   Odometry   │          │
│   └──────────┘    └──────────┘    └──────┬───────┘          │
│                                          │                   │
│   ┌──────────┐                    ┌──────▼───────┐          │
│   │   IMU    │───────────────────▶│ Sensor       │          │
│   │  Data    │                    │ Fusion       │          │
│   └──────────┘                    └──────┬───────┘          │
│                                          │                   │
│                                   ┌──────▼───────┐          │
│                                   │   Pose       │──▶ TF    │
│                                   │  Estimate    │          │
│                                   └──────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

```bash
# Install Isaac ROS Common
mkdir -p ~/workspaces/isaac_ros-dev/src
cd ~/workspaces/isaac_ros-dev/src

git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git

# Build in Docker (recommended)
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common
./scripts/run_dev.sh

# Inside container
cd /workspaces/isaac_ros-dev
colcon build --symlink-install
source install/setup.bash
```

## Using cuVSLAM with RealSense Camera

### Launch File

```python
# launch/humanoid_vslam.launch.py
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    # NITROS container for zero-copy GPU pipeline
    vslam_container = ComposableNodeContainer(
        name='vslam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # RealSense camera node
            ComposableNode(
                package='realsense2_camera',
                plugin='realsense2_camera::RealSenseNodeFactory',
                name='camera',
                parameters=[{
                    'enable_infra1': True,
                    'enable_infra2': True,
                    'enable_gyro': True,
                    'enable_accel': True,
                    'gyro_fps': 200,
                    'accel_fps': 200,
                    'infra_fps': 30,
                }],
            ),

            # cuVSLAM node
            ComposableNode(
                package='isaac_ros_visual_slam',
                plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
                name='visual_slam',
                parameters=[{
                    'denoise_input_images': True,
                    'rectified_images': True,
                    'enable_imu_fusion': True,
                    'gyro_noise_density': 0.00016,
                    'gyro_random_walk': 0.000022,
                    'accel_noise_density': 0.00186,
                    'accel_random_walk': 0.00046,
                    'calibration_frequency': 200.0,
                    'img_jitter_threshold_ms': 34.0,
                }],
                remappings=[
                    ('stereo_camera/left/image', '/camera/infra1/image_rect_raw'),
                    ('stereo_camera/left/camera_info', '/camera/infra1/camera_info'),
                    ('stereo_camera/right/image', '/camera/infra2/image_rect_raw'),
                    ('stereo_camera/right/camera_info', '/camera/infra2/camera_info'),
                    ('visual_slam/imu', '/camera/imu'),
                ],
            ),
        ],
    )

    return LaunchDescription([vslam_container])
```

## cuVSLAM Node Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enable_imu_fusion` | Fuse IMU data with visual odometry | `true` |
| `denoise_input_images` | Apply denoising to stereo images | `true` |
| `rectified_images` | Input images are already rectified | `true` |
| `enable_slam_visualization` | Publish visualization markers | `true` |
| `enable_observations_view` | Publish tracked features | `false` |
| `enable_landmarks_view` | Publish 3D landmarks | `false` |

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/visual_slam/tracking/odometry` | `nav_msgs/Odometry` | Robot pose and velocity |
| `/visual_slam/status` | `isaac_ros_visual_slam_interfaces/VisualSlamStatus` | Tracking status |
| `/tf` | `tf2_msgs/TFMessage` | Transform: odom → base_link |

## Integrating with Humanoid Control

### Pose Subscriber for Balance Control

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np

class HumanoidLocalizer(Node):
    def __init__(self):
        super().__init__('humanoid_localizer')

        self.pose_sub = self.create_subscription(
            Odometry,
            '/visual_slam/tracking/odometry',
            self.odometry_callback,
            10
        )

        # Store pose history for velocity estimation
        self.pose_history = []
        self.max_history = 10

    def odometry_callback(self, msg):
        # Extract position
        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        # Extract orientation (quaternion)
        orientation = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

        # Extract velocities
        linear_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])

        angular_vel = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])

        # Use for balance control
        self.update_balance_controller(
            position, orientation,
            linear_vel, angular_vel
        )

    def update_balance_controller(self, pos, orient, lin_vel, ang_vel):
        """Send pose to balance controller."""
        # Estimate body tilt from orientation
        # Adjust center of mass based on velocity
        # ... balance control logic
        pass
```

## Loop Closure and Map Saving

### Saving Maps for Re-localization

```bash
# Save current map
ros2 service call /visual_slam/save_map \
    isaac_ros_visual_slam_interfaces/srv/FilePath \
    "{file_path: '/maps/humanoid_lab.db'}"

# Load map for re-localization
ros2 service call /visual_slam/load_map_and_localize \
    isaac_ros_visual_slam_interfaces/srv/FilePath \
    "{file_path: '/maps/humanoid_lab.db'}"
```

## Performance on Different Platforms

| Platform | Resolution | FPS | Latency |
|----------|------------|-----|---------|
| **Jetson Orin Nano** | 640x480 | 30 | 33ms |
| **Jetson Orin NX** | 1280x720 | 60 | 16ms |
| **RTX 4070** | 1280x720 | 90 | 11ms |
| **RTX 4090** | 1920x1080 | 120 | 8ms |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Tracking lost** | Reduce movement speed, increase lighting |
| **High latency** | Use NITROS for zero-copy GPU pipeline |
| **Drift over time** | Enable loop closure, use IMU fusion |
| **Poor in textureless areas** | Add visual features or use LiDAR |

## Key Takeaways

1. **cuVSLAM** provides GPU-accelerated visual odometry for real-time localization
2. **IMU fusion** improves robustness during fast movements
3. **NITROS** enables zero-copy data transfer for minimal latency
4. **Map saving/loading** allows re-localization in known environments
5. **Integration with Nav2** enables autonomous navigation

---

*Next: Configure Nav2 for bipedal humanoid path planning.*
