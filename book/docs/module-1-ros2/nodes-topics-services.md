---
sidebar_position: 3
title: "Nodes, Topics, and Services"
---

# ROS 2 Communication: Nodes, Topics, and Services

## Overview

ROS 2 provides three primary communication patterns that form the backbone of robot software architecture. Understanding when to use each pattern is essential for building efficient humanoid robot systems.

## Nodes: The Building Blocks

A **node** is a single-purpose process that performs a specific task. In humanoid robotics, you might have separate nodes for:

- **Sensor processing** (camera, IMU, LiDAR)
- **Motion planning** (trajectory generation)
- **Balance control** (real-time stabilization)
- **Speech recognition** (voice commands)
- **AI inference** (VLA model execution)

### Creating a Node in Python (rclpy)

```python
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        self.get_logger().info('Humanoid Controller initialized')

        # Node is now running and can communicate

def main():
    rclpy.init()
    node = HumanoidController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Topics: Publish/Subscribe Communication

**Topics** enable one-to-many asynchronous communication. A publisher sends messages without knowing who (if anyone) is listening. Subscribers receive messages without knowing who sent them.

### When to Use Topics

| Use Case | Example |
|----------|---------|
| Streaming sensor data | Camera frames at 30 Hz |
| Continuous state updates | Joint positions at 100 Hz |
| Broadcasting information | Robot status to multiple listeners |
| Decoupled communication | Sensor doesn't need to know about consumers |

### Publisher Example

```python
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

class JointPublisher(Node):
    def __init__(self):
        super().__init__('joint_publisher')

        # Create publisher for joint states
        self.publisher = self.create_publisher(
            JointState,
            '/humanoid/joint_states',
            10  # Queue size
        )

        # Publish at 100 Hz
        self.timer = self.create_timer(0.01, self.publish_joints)

    def publish_joints(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['hip_pitch', 'hip_roll', 'knee', 'ankle_pitch']
        msg.position = [0.0, 0.0, 0.5, 0.2]  # radians
        msg.velocity = [0.0, 0.0, 0.0, 0.0]
        self.publisher.publish(msg)
```

### Subscriber Example

```python
class JointSubscriber(Node):
    def __init__(self):
        super().__init__('joint_subscriber')

        self.subscription = self.create_subscription(
            JointState,
            '/humanoid/joint_states',
            self.joint_callback,
            10
        )

    def joint_callback(self, msg):
        self.get_logger().info(f'Received joint positions: {msg.position}')
```

## Services: Request/Response Communication

**Services** provide synchronous request/response communication. The client sends a request and waits for a response.

### When to Use Services

| Use Case | Example |
|----------|---------|
| One-time queries | Get current robot pose |
| Configuration changes | Set control mode |
| Trigger actions | Calibrate sensors |
| Computations | Calculate inverse kinematics |

### Service Definition (srv file)

```
# GetRobotPose.srv
---
geometry_msgs/Pose pose
bool success
string message
```

### Service Server

```python
from humanoid_interfaces.srv import GetRobotPose
from geometry_msgs.msg import Pose

class PoseService(Node):
    def __init__(self):
        super().__init__('pose_service')

        self.srv = self.create_service(
            GetRobotPose,
            '/humanoid/get_pose',
            self.get_pose_callback
        )

    def get_pose_callback(self, request, response):
        response.pose = Pose()
        response.pose.position.x = 1.0
        response.pose.position.y = 2.0
        response.pose.position.z = 0.9  # Standing height
        response.success = True
        response.message = 'Pose retrieved successfully'
        return response
```

### Service Client

```python
class PoseClient(Node):
    def __init__(self):
        super().__init__('pose_client')
        self.client = self.create_client(GetRobotPose, '/humanoid/get_pose')

    def get_current_pose(self):
        request = GetRobotPose.Request()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
```

## Actions: Long-Running Tasks with Feedback

**Actions** are for tasks that take time and need progress feedback. They support:
- Goal submission
- Progress feedback
- Result on completion
- Cancellation

### When to Use Actions

| Use Case | Example |
|----------|---------|
| Navigation | Walk to waypoint |
| Manipulation | Pick up object |
| Complex motions | Execute dance sequence |
| Any task > 1 second | Tasks that might need cancellation |

### Action Definition (action file)

```
# WalkToGoal.action
# Goal
geometry_msgs/Pose target_pose
float32 max_velocity
---
# Result
bool success
float32 final_distance
---
# Feedback
float32 distance_remaining
float32 current_velocity
```

### Action Server

```python
from rclpy.action import ActionServer
from humanoid_interfaces.action import WalkToGoal

class WalkActionServer(Node):
    def __init__(self):
        super().__init__('walk_action_server')

        self._action_server = ActionServer(
            self,
            WalkToGoal,
            '/humanoid/walk_to_goal',
            self.execute_callback
        )

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing walk goal...')

        feedback = WalkToGoal.Feedback()
        distance = 10.0  # Starting distance

        while distance > 0.1:
            # Simulate walking
            distance -= 0.5
            feedback.distance_remaining = distance
            feedback.current_velocity = 0.5
            goal_handle.publish_feedback(feedback)
            await asyncio.sleep(0.1)

        goal_handle.succeed()

        result = WalkToGoal.Result()
        result.success = True
        result.final_distance = distance
        return result
```

## Communication Pattern Summary

```
┌─────────────────────────────────────────────────────────────┐
│              Communication Pattern Selection                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Streaming Data?  ──Yes──▶  TOPICS (pub/sub)               │
│        │                                                    │
│        No                                                   │
│        │                                                    │
│  Quick Query?  ──Yes──▶  SERVICES (request/response)       │
│        │                                                    │
│        No                                                   │
│        │                                                    │
│  Long Task + Feedback?  ──Yes──▶  ACTIONS                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Quality of Service (QoS)

ROS 2 introduces QoS profiles for fine-grained control:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# For sensor data (best effort, latest only)
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

# For commands (reliable delivery)
command_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_ALL,
    depth=10
)
```

## Humanoid Robot Node Architecture

A typical humanoid system might use all three patterns:

```
┌─────────────────────────────────────────────────────────────┐
│                  Humanoid Node Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   TOPICS (continuous):                                      │
│   • /joint_states (100 Hz)                                 │
│   • /imu/data (200 Hz)                                     │
│   • /camera/image_raw (30 Hz)                              │
│   • /cmd_vel (50 Hz)                                       │
│                                                             │
│   SERVICES (on-demand):                                     │
│   • /get_robot_state                                       │
│   • /set_control_mode                                      │
│   • /calibrate_imu                                         │
│                                                             │
│   ACTIONS (long-running):                                   │
│   • /walk_to_goal                                          │
│   • /pick_object                                           │
│   • /execute_gesture                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Takeaways

1. **Nodes** = Single-purpose processes
2. **Topics** = Streaming, decoupled, one-to-many
3. **Services** = Synchronous request/response
4. **Actions** = Long tasks with feedback and cancellation
5. **QoS** = Control reliability and history for each connection

---

*Next: Learn how to bridge Python AI agents to ROS 2 controllers using rclpy.*
