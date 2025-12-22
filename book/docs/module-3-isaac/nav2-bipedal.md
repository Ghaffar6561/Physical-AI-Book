---
sidebar_position: 4
title: "Nav2 for Bipedal Robots"
---

# Nav2 Path Planning for Bipedal Humanoids

## Overview

**Nav2** (Navigation 2) is the ROS 2 navigation stack that enables autonomous robot navigation. For bipedal humanoids, Nav2 requires special configuration to account for the unique locomotion characteristics—step height constraints, balance requirements, and the ability to step over obstacles that wheeled robots cannot.

## Bipedal Navigation Challenges

| Challenge | Wheeled Robots | Bipedal Humanoids |
|-----------|----------------|-------------------|
| **Obstacle traversal** | Go around | Can step over/on |
| **Stairs** | Cannot climb | Can ascend/descend |
| **Narrow passages** | Width limited | Can turn in place |
| **Uneven terrain** | Limited | Adaptable |
| **Speed** | Fast | Slower, stable |

## Nav2 Architecture for Humanoids

```
┌─────────────────────────────────────────────────────────────┐
│                Nav2 Bipedal Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐    ┌──────────────┐                      │
│   │   Costmap    │───▶│   Planner    │                      │
│   │   Server     │    │   Server     │                      │
│   │ (2D + 3D)    │    │ (Footstep)   │                      │
│   └──────────────┘    └──────┬───────┘                      │
│                              │                               │
│   ┌──────────────┐    ┌──────▼───────┐                      │
│   │  Behavior    │◀───│  Controller  │                      │
│   │   Server     │    │   Server     │                      │
│   │ (Recovery)   │    │  (Gait)      │                      │
│   └──────────────┘    └──────┬───────┘                      │
│                              │                               │
│                       ┌──────▼───────┐                      │
│                       │  Locomotion  │                      │
│                       │  Controller  │                      │
│                       └──────────────┘                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Costmap Configuration for Humanoids

### 3D Costmap for Step Planning

```yaml
# config/nav2_costmap.yaml
global_costmap:
  ros__parameters:
    update_frequency: 5.0
    publish_frequency: 2.0
    global_frame: map
    robot_base_frame: base_link

    # Use 3D voxel layer for humanoids
    plugins: ["static_layer", "voxel_layer", "inflation_layer"]

    voxel_layer:
      plugin: "nav2_costmap_2d::VoxelLayer"
      enabled: True
      publish_voxel_map: True
      origin_z: 0.0
      z_resolution: 0.05        # 5cm vertical resolution
      z_voxels: 40              # Up to 2m height
      max_obstacle_height: 2.0
      mark_threshold: 0
      observation_sources: rgbd_sensor lidar

      # RealSense depth camera
      rgbd_sensor:
        topic: /camera/depth/points
        sensor_frame: camera_link
        observation_persistence: 0.0
        expected_update_rate: 30.0
        data_type: PointCloud2
        clearing: True
        marking: True
        max_obstacle_height: 2.0
        min_obstacle_height: 0.1  # Ignore ground

    # Humanoid-specific inflation
    inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      cost_scaling_factor: 3.0
      inflation_radius: 0.35     # Humanoid footprint radius
```

### Footstep-Aware Planning Layer

```yaml
# Custom layer for step height analysis
step_layer:
  plugin: "humanoid_nav/StepHeightLayer"
  enabled: True
  max_step_height: 0.25        # 25cm max step
  min_step_height: 0.03        # 3cm min to consider
  step_cost_multiplier: 2.0    # Penalize stepping
  stair_detection: True
```

## Footstep Planner

For humanoids, we need a **footstep planner** instead of a simple path planner:

```python
# humanoid_nav/footstep_planner.py
import numpy as np
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
from humanoid_interfaces.msg import FootstepPlan, Footstep

class FootstepPlanner:
    """Plans individual footstep placements for bipedal walking."""

    def __init__(self):
        self.step_length = 0.30      # meters
        self.step_width = 0.20       # meters
        self.max_step_height = 0.25  # meters

    def plan_footsteps(self, start_pose, goal_pose, costmap):
        """Generate footstep sequence from start to goal."""
        footsteps = []

        # Get path from Nav2 planner
        path = self.get_nav2_path(start_pose, goal_pose)

        # Convert path waypoints to footsteps
        current_foot = 'left'  # Alternating feet
        current_pos = np.array([start_pose.position.x, start_pose.position.y])

        for waypoint in path.poses:
            target = np.array([waypoint.pose.position.x, waypoint.pose.position.y])
            direction = target - current_pos
            distance = np.linalg.norm(direction)

            if distance > 0.01:  # Minimum step
                direction = direction / distance

                # Calculate steps needed
                num_steps = int(np.ceil(distance / self.step_length))

                for i in range(num_steps):
                    # Alternating foot placement
                    lateral_offset = self.step_width / 2
                    if current_foot == 'left':
                        lateral_offset = -lateral_offset

                    # Calculate footstep position
                    step_pos = current_pos + direction * min(self.step_length, distance)

                    # Check terrain height from costmap
                    terrain_height = self.get_terrain_height(step_pos, costmap)

                    footstep = Footstep()
                    footstep.foot = current_foot
                    footstep.pose.position.x = step_pos[0]
                    footstep.pose.position.y = step_pos[1] + lateral_offset
                    footstep.pose.position.z = terrain_height

                    footsteps.append(footstep)

                    # Switch feet
                    current_foot = 'right' if current_foot == 'left' else 'left'
                    current_pos = step_pos

        return FootstepPlan(footsteps=footsteps)

    def get_terrain_height(self, position, costmap):
        """Query 3D costmap for terrain elevation."""
        # Implementation depends on costmap representation
        return 0.0  # Ground level default
```

## Gait Controller Integration

```python
# humanoid_control/gait_controller.py
from rclpy.node import Node
from humanoid_interfaces.msg import FootstepPlan, GaitCommand
from sensor_msgs.msg import JointState

class GaitController(Node):
    """Converts footstep plans to joint trajectories."""

    def __init__(self):
        super().__init__('gait_controller')

        self.footstep_sub = self.create_subscription(
            FootstepPlan,
            '/humanoid/footstep_plan',
            self.execute_footsteps,
            10
        )

        self.joint_pub = self.create_publisher(
            JointState,
            '/humanoid/joint_commands',
            10
        )

        # Gait parameters
        self.step_duration = 0.5  # seconds per step
        self.double_support_ratio = 0.2  # 20% of step in double support

    def execute_footsteps(self, plan):
        """Execute footstep plan using inverse kinematics."""
        for i, footstep in enumerate(plan.footsteps):
            self.get_logger().info(f'Executing step {i+1}/{len(plan.footsteps)}')

            # Generate swing trajectory
            trajectory = self.compute_swing_trajectory(footstep)

            # Execute trajectory
            for joint_state in trajectory:
                self.joint_pub.publish(joint_state)
                self.get_clock().sleep_for(Duration(seconds=0.01))

    def compute_swing_trajectory(self, target_footstep):
        """Compute joint angles for swing phase."""
        trajectory = []

        # Lift phase
        lift_height = 0.05  # 5cm foot lift

        # Use inverse kinematics to compute joint angles
        # for each point along the swing trajectory

        return trajectory
```

## Nav2 Launch for Humanoids

```python
# launch/humanoid_nav2.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('humanoid_navigation')

    nav2_params = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')

    return LaunchDescription([
        # Costmap server
        Node(
            package='nav2_costmap_2d',
            executable='nav2_costmap_2d',
            name='global_costmap',
            parameters=[nav2_params],
        ),

        # Planner server (with footstep planning)
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            parameters=[nav2_params],
        ),

        # Behavior server
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            parameters=[nav2_params],
        ),

        # Custom gait controller
        Node(
            package='humanoid_control',
            executable='gait_controller',
            name='gait_controller',
            parameters=[nav2_params],
        ),

        # Lifecycle manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            parameters=[{
                'autostart': True,
                'node_names': [
                    'global_costmap',
                    'planner_server',
                    'behavior_server',
                    'gait_controller',
                ]
            }],
        ),
    ])
```

## Nav2 Parameters for Humanoids

```yaml
# config/nav2_params.yaml
planner_server:
  ros__parameters:
    expected_planner_frequency: 2.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_smac_planner/SmacPlanner2D"
      tolerance: 0.25
      allow_unknown: true
      max_iterations: 1000000
      max_planning_time: 5.0

behavior_server:
  ros__parameters:
    costmap_topic: global_costmap/costmap_raw
    footprint_topic: global_costmap/published_footprint
    behavior_plugins: ["backup", "wait", "spin"]

    # Humanoid-specific recovery behaviors
    backup:
      plugin: "nav2_behaviors/BackUp"
      enabled: False  # Humanoids don't backup easily

    wait:
      plugin: "nav2_behaviors/Wait"
      enabled: True

    spin:
      plugin: "nav2_behaviors/Spin"
      enabled: True   # Turn in place is natural for bipeds
```

## Stair Navigation

Humanoids can navigate stairs, requiring special handling:

```python
class StairNavigator:
    """Specialized navigation for stair climbing."""

    def __init__(self):
        self.stair_detector = StairDetector()
        self.step_height = 0.18  # Standard stair height
        self.step_depth = 0.28   # Standard stair depth

    def detect_and_climb_stairs(self, point_cloud):
        stairs = self.stair_detector.detect(point_cloud)

        if stairs.detected:
            # Switch to stair climbing gait
            footstep_plan = self.plan_stair_footsteps(stairs)
            return footstep_plan

        return None
```

## Key Takeaways

1. **3D costmaps** enable humanoids to reason about step height
2. **Footstep planners** replace simple path planners for bipeds
3. **Gait controllers** translate footsteps to joint trajectories
4. **Stair navigation** is uniquely possible for humanoids
5. **Recovery behaviors** differ from wheeled robots (no backup)

---

*Next: Learn humanoid kinematics and dynamics for locomotion control.*
