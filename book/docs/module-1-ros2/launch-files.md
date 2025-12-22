---
sidebar_position: 5
title: "Launch Files and Parameters"
---

# Launch Files and Parameter Management

## Overview

Real humanoid robot systems require launching dozens of nodes simultaneously with specific configurations. ROS 2 **launch files** orchestrate this complexity, while **parameters** allow runtime configuration without code changes.

## Launch File Basics

Launch files in ROS 2 are Python scripts that define which nodes to start and how to configure them.

### Simple Launch File

```python
# launch/humanoid_bringup.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Start joint state publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen'
        ),

        # Start robot state publisher (for TF)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_urdf}],
            output='screen'
        ),

        # Start our AI controller
        Node(
            package='humanoid_ai_agents',
            executable='vision_agent',
            name='vision_agent',
            output='screen'
        ),
    ])
```

## Parameters and Configuration

### Loading Parameters from YAML

```yaml
# config/humanoid_params.yaml
humanoid_controller:
  ros__parameters:
    # Control parameters
    control_frequency: 100.0  # Hz
    max_joint_velocity: 2.0   # rad/s
    max_joint_torque: 50.0    # Nm

    # AI model parameters
    model_path: "/models/vla_model.pt"
    inference_device: "cuda"
    confidence_threshold: 0.7

    # Safety limits
    workspace_limits:
      x_min: -2.0
      x_max: 2.0
      y_min: -2.0
      y_max: 2.0
      z_min: 0.0
      z_max: 2.5
```

### Launch File with Parameters

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('humanoid_control')

    # Path to parameter file
    params_file = os.path.join(pkg_dir, 'config', 'humanoid_params.yaml')

    return LaunchDescription([
        Node(
            package='humanoid_control',
            executable='humanoid_controller',
            name='humanoid_controller',
            parameters=[params_file],
            output='screen'
        ),
    ])
```

### Accessing Parameters in Node

```python
class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Declare and get parameters
        self.declare_parameter('control_frequency', 100.0)
        self.declare_parameter('max_joint_velocity', 2.0)
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.7)

        # Read parameter values
        self.control_freq = self.get_parameter('control_frequency').value
        self.max_velocity = self.get_parameter('max_joint_velocity').value
        self.model_path = self.get_parameter('model_path').value

        # Set up control loop at specified frequency
        period = 1.0 / self.control_freq
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info(
            f'Controller running at {self.control_freq} Hz'
        )
```

## Advanced Launch Patterns

### Conditional Launch with Arguments

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim = DeclareLaunchArgument(
        'use_sim',
        default_value='true',
        description='Use simulation instead of real robot'
    )

    enable_ai = DeclareLaunchArgument(
        'enable_ai',
        default_value='true',
        description='Enable AI perception nodes'
    )

    return LaunchDescription([
        use_sim,
        enable_ai,

        # Gazebo (only in simulation)
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'humanoid', '-file', urdf_path],
            condition=IfCondition(LaunchConfiguration('use_sim'))
        ),

        # AI nodes (optional)
        Node(
            package='humanoid_ai_agents',
            executable='vision_agent',
            condition=IfCondition(LaunchConfiguration('enable_ai'))
        ),
    ])
```

### Including Other Launch Files

```python
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Include Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/gazebo.launch.py'
        ]),
        launch_arguments={'world': world_path}.items()
    )

    # Include robot-specific launch
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('humanoid_description'),
            '/launch/robot.launch.py'
        ])
    )

    return LaunchDescription([
        gazebo_launch,
        robot_launch,
        # Additional nodes...
    ])
```

## Complete Humanoid System Launch

```python
# launch/full_humanoid_system.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    # Package directories
    pkg_control = get_package_share_directory('humanoid_control')
    pkg_perception = get_package_share_directory('humanoid_perception')
    pkg_ai = get_package_share_directory('humanoid_ai_agents')

    # Launch arguments
    robot_name = DeclareLaunchArgument('robot_name', default_value='humanoid_1')
    use_sim = DeclareLaunchArgument('use_sim', default_value='true')
    enable_voice = DeclareLaunchArgument('enable_voice', default_value='true')

    # Parameter files
    control_params = os.path.join(pkg_control, 'config', 'control.yaml')
    perception_params = os.path.join(pkg_perception, 'config', 'perception.yaml')
    ai_params = os.path.join(pkg_ai, 'config', 'ai_models.yaml')

    # Group nodes under robot namespace
    robot_nodes = GroupAction([
        PushRosNamespace(LaunchConfiguration('robot_name')),

        # Core control nodes
        Node(
            package='humanoid_control',
            executable='joint_controller',
            parameters=[control_params],
        ),
        Node(
            package='humanoid_control',
            executable='balance_controller',
            parameters=[control_params],
        ),

        # Perception nodes
        Node(
            package='humanoid_perception',
            executable='camera_processor',
            parameters=[perception_params],
        ),
        Node(
            package='humanoid_perception',
            executable='lidar_processor',
            parameters=[perception_params],
        ),

        # AI nodes
        Node(
            package='humanoid_ai_agents',
            executable='vla_controller',
            parameters=[ai_params],
        ),

        # Voice interface (optional)
        Node(
            package='humanoid_ai_agents',
            executable='whisper_node',
            parameters=[ai_params],
            condition=IfCondition(LaunchConfiguration('enable_voice'))
        ),
    ])

    return LaunchDescription([
        robot_name,
        use_sim,
        enable_voice,
        robot_nodes,
    ])
```

## Running Launch Files

```bash
# Basic launch
ros2 launch humanoid_control humanoid_bringup.launch.py

# With arguments
ros2 launch humanoid_control full_humanoid_system.launch.py \
    robot_name:=humanoid_1 \
    use_sim:=true \
    enable_voice:=true

# Override parameters at runtime
ros2 launch humanoid_control humanoid_bringup.launch.py \
    --ros-args -p control_frequency:=200.0
```

## Dynamic Parameter Updates

```python
# Update parameter at runtime
ros2 param set /humanoid_controller control_frequency 200.0

# Get current parameter value
ros2 param get /humanoid_controller control_frequency

# List all parameters
ros2 param list /humanoid_controller
```

### Handling Dynamic Updates in Node

```python
from rcl_interfaces.msg import SetParametersResult

class DynamicController(Node):
    def __init__(self):
        super().__init__('dynamic_controller')

        self.declare_parameter('gain', 1.0)
        self.gain = self.get_parameter('gain').value

        # Register callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'gain':
                self.gain = param.value
                self.get_logger().info(f'Gain updated to {self.gain}')

        return SetParametersResult(successful=True)
```

## Key Takeaways

| Concept | Purpose |
|---------|---------|
| **Launch files** | Orchestrate multi-node systems |
| **Parameters** | Configure nodes without code changes |
| **Launch arguments** | Customize launch at runtime |
| **Namespaces** | Isolate multiple robots |
| **Conditions** | Enable/disable nodes dynamically |
| **Dynamic params** | Update configuration at runtime |

---

*Next: Define your humanoid robot structure using URDF.*
