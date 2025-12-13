# ROS 2 Introduction

## What is ROS 2?

ROS 2 (Robot Operating System 2) is the standard middleware for building robotic systems. It provides:

- **Communication infrastructure** — Publish/subscribe messaging between processes
- **Standard tools** — Command-line utilities, debugging, introspection
- **Hardware abstraction** — Write code once, run on different robots
- **Ecosystem** — Libraries for perception, control, planning, simulation

## ROS 2 Core Concepts

### Nodes

A **node** is a process that performs computation. It might:
- Read sensor data and publish it
- Subscribe to sensor data and process it
- Execute control decisions
- Manage communication

Nodes are independent—they can run on different machines and restart without crashing the whole system.

### Topics

A **topic** is a named bus for one-way communication. Multiple nodes can:
- **Publish** messages to a topic
- **Subscribe** to a topic and receive messages

This is asynchronous—publishers don't care if anyone is listening.

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher = self.create_publisher(Point, 'object_position', 10)
        
    def publish_position(self, x, y, z):
        msg = Point(x=float(x), y=float(y), z=float(z))
        self.publisher.publish(msg)
```

### Services

A **service** is synchronous request-response communication. One node **calls** a service and waits for a response from another node.

Use services for:
- Configuration queries
- Computations that block briefly
- Request-response patterns

Use topics for:
- Continuous sensor streams
- Fire-and-forget messages

### Actions

An **action** is for long-running tasks that take time and can be cancelled. Examples:
- "Move the arm to position X" — takes 2 seconds, can fail
- "Navigate to location Y" — takes 30 seconds, needs feedback

Actions provide:
- Progress feedback (% complete)
- Ability to cancel mid-execution
- Success/failure status

## Execution

Nodes run in a distributed system:

```
Machine 1           Machine 2
┌─────────────┐     ┌─────────────┐
│ Sensor Node │──→ Topic: sensor_data
│             │     ↓
└─────────────┘     ┌─────────────┐
                    │Processing   │
                    │   Node      │
                    └─────────────┘
                            │
                            ↓
                    Topic: processed_data
                            ↓
                    ┌─────────────┐
                    │ Control     │──→ Motors
                    │ Node        │
                    └─────────────┘
```

This is powerful because:
- **Modularity** — Each node does one thing well
- **Reusability** — Standard interfaces (topics, services)
- **Distributedness** — Nodes can run on different machines
- **Robustness** — One node crashing doesn't crash the system

## Your First ROS 2 Program

Coming in the exercises section.

---

**Next**: [Exercises](exercises.md)
