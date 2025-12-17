# ROS 2 Introduction

## What is ROS 2?

ROS 2 (Robot Operating System 2) is the standard middleware for building robotic systems. It provides:

- **Communication infrastructure** — Publish/subscribe messaging between processes
- **Standard tools** — Command-line utilities, debugging, introspection
- **Hardware abstraction** — Write code once, run on different robots
- **Ecosystem** — Libraries for perception, control, planning, simulation

ROS 2 is the software foundation for embodied intelligence. Without it, coordinating sensor input, perception, planning, and motor control becomes a nightmare of synchronization, buffering, and timing issues.

## ROS 2 Core Concepts

### Nodes: Independent Processes

A **node** is a process that performs computation. It might:
- **Sense**: Read sensor data and publish it (e.g., camera images, LiDAR scans)
- **Process**: Subscribe to sensor data and compute something useful (e.g., object detection)
- **Plan**: Decide what to do based on perception (e.g., "navigate to the table")
- **Control**: Execute decisions by commanding motors (e.g., joint angles)

**Key insight**: Nodes are **independent**. They:
- Can run on different machines
- Can restart without crashing the whole system
- Can be added/removed dynamically
- Have no shared state

This independence is crucial for **robustness**. If one node crashes (e.g., object detection times out), it doesn't crash the robot.

### Topics: One-Way, Asynchronous Communication

A **topic** is a named bus for one-way communication. Multiple nodes can:
- **Publish** messages to a topic (one publisher or many)
- **Subscribe** to a topic and receive messages (one receiver or many)

This is **asynchronous**—publishers don't care if anyone is listening. Publishers and subscribers are decoupled.

**Example: Robot Vision**

```python
# Publisher node: Publishes camera images
class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_node')
        # Publish images to 'camera/image' topic
        self.publisher = self.create_publisher(Image, 'camera/image', 10)

    def publish_frame(self, frame):
        # Convert frame to ROS 2 message and publish
        msg = self.convert_to_image_msg(frame)
        self.publisher.publish(msg)

# Subscriber node: Receives and processes images
class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        # Subscribe to 'camera/image' topic
        self.subscription = self.create_subscription(
            Image, 'camera/image', self.detect_objects, 10
        )

    def detect_objects(self, msg):
        # Called whenever a new image arrives
        detections = self.neural_network(msg)
        # Publish results to another topic
        self.publish_detections(detections)
```

**Why topics matter for embodied intelligence**:
- **Real-time**: Can process sensor data as it arrives
- **Flexible**: Any number of publishers/subscribers can join
- **Decoupled**: Each component focuses on one job

### Services: Request-Response Communication

A **service** is synchronous request-response communication. One node **calls** a service and **waits** for a response.

**When to use services**:
- Configuration queries ("get robot state")
- Computations that must complete ("compute inverse kinematics")
- Request-response patterns ("turn on gripper")

**When NOT to use services**:
- Continuous sensor streams (use topics)
- High-frequency updates (use topics)
- Fire-and-forget commands (use topics)

**Example: Gripper Control**

```python
# Server: Provides gripper control service
class GripperServer(Node):
    def __init__(self):
        super().__init__('gripper_server')
        self.srv = self.create_service(
            GripperCommand, 'control_gripper', self.gripper_callback
        )

    def gripper_callback(self, request):
        # Close or open gripper based on request
        if request.action == "close":
            self.close_gripper()
        else:
            self.open_gripper()
        return GripperCommandResponse(success=True)

# Client: Calls gripper control service
class Manipulator(Node):
    def __init__(self):
        super().__init__('manipulator')
        self.client = self.create_client(GripperCommand, 'control_gripper')

    def grasp_object(self):
        # Send request and wait for response
        request = GripperCommand()
        request.action = "close"
        response = self.client.call(request)
        return response.success
```

### Actions: Long-Running Tasks with Feedback

An **action** is for tasks that take time and need feedback. Examples:
- "Move the arm to position X" — takes 2 seconds, can fail
- "Navigate to location Y" — takes 30 seconds, robot should report progress
- "Pick up the red ball" — complex multi-step, should send feedback

Actions provide:
- **Goal**: What the action should accomplish
- **Feedback**: Progress updates while running
- **Result**: Success/failure with final state
- **Cancellation**: Stop the action mid-execution

**Example: Robot Navigation**

```python
# Server: Provides navigation action
class NavigationServer(Node):
    def __init__(self):
        super().__init__('navigation_server')
        self._action_server = ActionServer(
            self, NavigateToPose, 'navigate_to_pose',
            self.execute_callback
        )

    def execute_callback(self, goal_handle):
        # Long-running computation: navigate to pose
        while not self.reached_goal():
            # Send feedback: current progress
            feedback = NavigateToPose.Feedback()
            feedback.current_pose = self.get_current_pose()
            goal_handle.publish_feedback(feedback)
            time.sleep(0.1)

        # Action completed successfully
        goal_handle.succeed()
        return NavigateToPose.Result()

# Client: Requests navigation
class Planner(Node):
    def __init__(self):
        super().__init__('planner')
        self._action_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )

    def navigate(self, target_pose):
        goal_msg = NavigateToPose.Goal()
        goal_msg.target = target_pose

        # Send goal and wait for result
        future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        return future.result()

    def feedback_callback(self, feedback_msg):
        # Called repeatedly as robot moves
        print(f"Robot at: {feedback_msg.feedback.current_pose}")
```

## Execution Model: Distributed Computing

Nodes run independently and communicate through topics/services:

```
Computer 1               Computer 2               Computer 3
┌──────────────┐         ┌──────────────┐        ┌──────────────┐
│  Sensor Node │         │ Perception   │        │  Control     │
│ (camera,     │──────→  │  Node        │──────→ │  Node        │──→ Motors
│  lidar)      │ Topics  │ (object      │Topics  │ (IK, traj.)  │
└──────────────┘         │  detection)  │        └──────────────┘
                         └──────────────┘

Connected via ROS 2 middleware (DDS)
Can run on local machine or remote network
```

**Why this distributed architecture matters**:
1. **Modularity** — Each node does one thing well
2. **Reusability** — Standard interfaces (topics, services, actions)
3. **Distributedness** — Nodes can run on different computers
4. **Robustness** — One node crashing doesn't crash the whole system
5. **Performance** — Can balance computation across hardware
6. **Testability** — Can test nodes independently

## Your First ROS 2 Program

### Publisher Example

Here's the simplest ROS 2 publisher node. It publishes "Hello, ROS 2!" every 0.5 seconds:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello, ROS 2!'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

def main():
    rclpy.init()
    publisher = MinimalPublisher()
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Example

Here's a minimal subscriber that listens to the publisher above:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String, 'topic', self.listener_callback, 10
        )

    def listener_callback(self, msg):
        # Called whenever a message arrives
        self.get_logger().info(f'I heard: "{msg.data}"')

def main():
    rclpy.init()
    subscriber = MinimalSubscriber()
    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Running These Examples

In terminal 1 (run publisher):
```bash
python minimal_publisher.py
# Output:
#   Publishing: Hello, ROS 2!
#   Publishing: Hello, ROS 2!
```

In terminal 2 (run subscriber):
```bash
python minimal_subscriber.py
# Output:
#   I heard: "Hello, ROS 2!"
#   I heard: "Hello, ROS 2!"
```

The key insight: **Publisher and subscriber are completely decoupled**. The publisher doesn't know or care if anyone is listening. The subscriber automatically receives all messages.

## Key Takeaways

| Concept | Use Case | Synchronous? | Multiple Publishers? |
|---------|----------|--------------|----------------------|
| **Topic** | Sensor streams, continuous data | No (async) | Yes |
| **Service** | Requests, queries, blocking calls | Yes | No |
| **Action** | Long-running tasks, navigation | Semi (goal + async feedback) | No |

## Summary

ROS 2 gives you:
- **Nodes**: Independent processes that do one thing well
- **Topics**: Efficient one-way messaging for continuous data
- **Services**: Synchronous request-response for queries
- **Actions**: Long-running tasks with progress feedback
- **Distributed computing**: Scale across multiple machines

These primitives are the building blocks for **embodied intelligence**. They let you build complex robot systems from simple, composable pieces.

---

**Next**: [Exercises](exercises.md) — Design your own ROS 2 system!
