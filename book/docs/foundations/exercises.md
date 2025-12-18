# Module 1: Exercises & Solutions

These exercises reinforce the key concepts from Module 1: embodied intelligence, perception-action loops, and ROS 2 fundamentals.

**Instructions**: Try each exercise first, then compare with the solution. Don't look at solutions until you've attempted the exercise!

---

## Exercise 1: Explain Embodied Intelligence

### Objective
Articulate the key differences between digital AI and physical AI. Demonstrate understanding of why robots are fundamentally different from software systems.

### Task
In your own words (200-300 words), explain:
1. **What is embodied intelligence?** (Not just "having a body," but what role the body plays in intelligence)
2. **How is a robot grasping an object different from an image classifier recognizing an object?**
3. **Why does millisecond timing matter in robotics but not in batch processing?**

### Success Criteria
- ✅ Define embodied intelligence clearly (not just "embodied = has a body")
- ✅ Give a concrete example showing the difference (not just abstract)
- ✅ Explain the real-time constraint and why it matters

### Hint
Think about feedback loops. What happens after a classifier recognizes "cat"? What happens after a robot perceives an object and grasps it?

### Solution

**What is embodied intelligence?**

Embodied intelligence is the principle that intelligence arises from the **interaction** between an agent and its **physical environment** through sensors and actuators. It's not just "having a body"—it's that the body and environment shape what intelligence means. A disembodied algorithm that processes images has no feedback from reality. A robot that grasps an object gets **immediate feedback**: did it succeed? Did the object slip? This feedback drives learning and adaptation.

Key insight: Embodied agents must think in **time** and **physics**, not just probability distributions.

**Robot Grasping vs. Image Classification**

Image classifier:
```
Image (pixels) → Neural network → Label ("cat") → Done
                                  (no feedback)
```

Robot grasping:
```
Perception (camera, force sensor) → Planning (where to grasp?) → Motor commands
     ↓ (feedback)
Did object slip? Adjust grip force
Did grasp succeed? Try again or report failure
       ↓ (loop continues)
```

The classifier makes one decision. The robot must **continuously adjust** based on sensory feedback. The timing of this feedback loop is **critical**. If gripper feedback arrives 500ms late, the robot drops the object.

**Why Timing Matters**

Batch processing (image classification):
- Process 1000 images offline
- Latency doesn't matter
- Accuracy is the metric

Robotics (real-time control):
- Robot walking at 1 m/s needs to process sensors and update motors every ~50-100ms
- If perception takes 5 seconds (typical for deep learning on CPU), the robot is blind mid-step
- Timing is **not optional**—it's safety-critical

Real-time constraint: A robot executing a task must maintain **closed-loop feedback** at sufficient frequency to keep up with the task dynamics.

---

## Exercise 2: Perception-Action Loop Design

### Objective
Design a perception-action loop for a real robot task. Understand how sensors, perception, planning, and control connect.

### Task
Choose one task and design its perception-action loop:
- A humanoid robot picking up a coffee cup
- A wheeled robot navigating through a crowded hallway
- A robot arm reaching to grasp an object on a moving conveyor belt

For your chosen task, specify:
1. **Sensors** — What does the robot sense? (cameras, force sensors, joint encoders, etc.)
2. **Perception** — What algorithms process sensor data? (object detection, localization, force estimation)
3. **Planning** — What decision is made? (where to move, how to grasp, whether to retry)
4. **Control** — What motor commands result? (joint angles, gripper force, wheel velocities)
5. **Feedback** — How does the result inform the next action? (did grasp succeed? did object slip?)

Create a diagram showing the flow and feedback loop.

### Success Criteria
- ✅ All components (sense, perceive, plan, control, feedback) are present
- ✅ Feedback loop is clearly shown
- ✅ Realistic for the chosen task
- ✅ Explains WHY feedback is necessary

### Solution: Robot Picking Up a Coffee Cup

```
PERCEPTION PHASE
                Sensors:
                - RGB-D camera: sees cup location, shape
                - Proprioceptors: know arm position
                - Force/torque sensors: detect contact
                        ↓
                Perception:
                - Detect cup in image (CNN)
                - Estimate 3D position (triangulation)
                - Compute grasp point (geometry)
                        ↓
PLANNING PHASE
                Plan:
                - Move arm above cup (Cartesian path planning)
                - Open gripper
                - Move to grasp point
                        ↓
CONTROL PHASE
                Control:
                - Send joint angle commands (IK solver)
                - Monitor forces in real-time
                - Close gripper when force threshold reached
                        ↓
                Execute:
                - Arm moves, gripper closes
                        ↓
FEEDBACK PHASE (Critical!)
                Sensors feedback to loop:
                ✗ Cup still on table? → Retry grasp
                ✓ Object in gripper?  → Lift arm
                ✗ Object slipping?    → Increase force
                ✓ Grasp stable?       → Execute next action

Loop repeats continuously at 50-100 Hz
```

**Why feedback is necessary**:
- No two cups are identical (size, weight, surface friction)
- No two grasps are perfectly calculated
- The world is unpredictable (cup shifts, gripper slips)
- Without feedback, the robot would fail on the first unexpected variation

---

## Exercise 3: ROS 2 Publisher-Subscriber Design

### Objective
Design a simple ROS 2 system using nodes, topics, and the pub-sub pattern.

### Task
Design a ROS 2 system for a mobile robot that:
1. Has a sensor node publishing camera images
2. Has a perception node subscribing to images and publishing detected objects
3. Has a planning node subscribing to detected objects and publishing movement commands
4. Has a control node subscribing to commands and controlling motors

Specify:
- **Nodes**: Name each node and describe what it computes
- **Topics**: Name each topic, message type, and direction (publisher → topic → subscriber)
- **Why pub-sub?**: Why is this better than a single monolithic program?

### Success Criteria
- ✅ Clear node names that describe their function
- ✅ Topic names follow ROS 2 conventions (lowercase, underscores)
- ✅ Each message type is appropriate for the data
- ✅ Explains decoupling and modularity benefits

### Solution: Mobile Robot Vision System

**Nodes:**
1. `camera_node` — Reads camera, publishes frames
2. `object_detector` — Subscribes to frames, publishes detected objects
3. `path_planner` — Subscribes to objects, publishes navigation goals
4. `motor_controller` — Subscribes to goals, commands motors

**Topics & Data Flow:**
```
camera_node
    ↓
camera/image → Image messages (RGB frames, 30 Hz)
    ↓
object_detector
    ↓
perception/objects → ObjectArray messages (bounding boxes, class labels)
    ↓
path_planner
    ↓
planning/goal → Point messages (target location in map frame)
    ↓
motor_controller
    ↓
motors/velocity → Twist messages (linear & angular velocity commands)
```

**Why pub-sub (instead of monolithic program)?**

Monolithic (bad for robots):
```python
while True:
    frame = camera.read()
    objects = detector(frame)
    if objects:
        path = planner(objects)
        commands = control(path)
        motors.send(commands)
```
Problems:
- If detector crashes, whole robot stops
- Can't run detector on GPU and motors on CPU
- Can't test detector without motors
- Hard to switch detectors (new model)
- Timing becomes chaotic (detector slow → motors slow)

ROS 2 pub-sub (good for robots):
- **Decoupled**: Detector can crash without crashing motors
- **Distributed**: Run detector on GPU computer, motors on robot computer
- **Testable**: Test detector with recorded images, not live robot
- **Modular**: Swap detector (TensorFlow → ONNX) without touching motor control
- **Real-time**: Each node runs at its own frequency
  - Camera: 30 Hz (camera speed)
  - Detector: 10 Hz (GPU speed, OK to be slow)
  - Motors: 100 Hz (real-time control requirement)

---

## Exercise 4: ROS 2 Code: Your First Pub-Sub System

### Objective
Write a simple ROS 2 publisher and subscriber. Experience the decoupling directly.

### Task
Modify the minimal_publisher.py and minimal_subscriber.py examples from the ROS 2 Introduction to:

**Publisher**: Publish "Sensor: temperature=XX.X°C" every 1 second (simulate a temperature sensor)

**Subscriber**: When it receives a message, parse the temperature and print:
- If temp > 30°C: "WARNING: High temperature!"
- If temp < 10°C: "WARNING: Low temperature!"
- Otherwise: "Temperature OK: XX.X°C"

**Requirement**: Don't modify publisher to know about the subscriber. Keep them decoupled.

### Success Criteria
- ✅ Publisher sends temperature messages
- ✅ Subscriber parses and reacts differently based on values
- ✅ Can run in separate terminals without knowing about each other
- ✅ Works with the minimal examples from ros2-intro.md

### Solution

**Publisher (temperature_sensor.py):**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random

class TemperatureSensor(Node):
    def __init__(self):
        super().__init__('temperature_sensor')
        self.publisher_ = self.create_publisher(String, 'sensor/temperature', 10)
        self.timer = self.create_timer(1.0, self.publish_temperature)

    def publish_temperature(self):
        # Simulate temperature sensor (random between 5 and 35°C)
        temp = 5 + random.random() * 30
        msg = String()
        msg.data = f'Sensor: temperature={temp:.1f}°C'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

def main():
    rclpy.init()
    sensor = TemperatureSensor()
    try:
        rclpy.spin(sensor)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Subscriber (temperature_monitor.py):**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TemperatureMonitor(Node):
    def __init__(self):
        super().__init__('temperature_monitor')
        self.subscription = self.create_subscription(
            String, 'sensor/temperature', self.monitor_temperature, 10
        )

    def monitor_temperature(self, msg):
        # Parse message: "Sensor: temperature=XX.X°C"
        try:
            # Extract temperature value
            parts = msg.data.split('=')
            temp_str = parts[1].replace('°C', '').strip()
            temp = float(temp_str)

            # React based on temperature
            if temp > 30.0:
                self.get_logger().warning(f'HIGH TEMP: {temp:.1f}°C')
            elif temp < 10.0:
                self.get_logger().warning(f'LOW TEMP: {temp:.1f}°C')
            else:
                self.get_logger().info(f'Temperature OK: {temp:.1f}°C')
        except Exception as e:
            self.get_logger().error(f'Failed to parse: {msg.data} - {e}')

def main():
    rclpy.init()
    monitor = TemperatureMonitor()
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Key insight**: The sensor publisher doesn't know a monitor exists. The monitor doesn't know about the sensor. They're completely decoupled through the topic. You could add a logger node, a graphing node, a database node—all subscribing to the same topic, without changing the publisher.

This is the power of ROS 2.

---

## Further Reading

### On Embodied Intelligence
- Rodney Brooks: "Intelligence Without Representation" (foundational paper)
- Pieter Abbeel & Sergey Levine: "Embodied AI in Robotics" (modern perspective)
- Francisco Varela & Evan Thompson: "The Embodied Mind" (philosophical grounding)

### On ROS 2
- [ROS 2 Official Documentation](https://docs.ros.org/en/humble/)
- ROS 2 Tutorials: Beginner Publisher/Subscriber
- MoveIt2 Documentation (for planning and control)

### On Robotics & Real-Time Systems
- Richard Sutton & Andrew Barto: "Reinforcement Learning" (Chapter 3 on MDPs)
- John Craig: "Introduction to Robotics" (kinematics and dynamics)

---

## Challenge Exercises (Optional)

### Challenge 1: Multi-Sensor Fusion
Design a ROS 2 system where:
- Two camera nodes publish images from different angles
- A detection node subscribes to both and merges detections
- Discuss: When would this be better than a single camera?

### Challenge 2: Real-Time Constraint Analysis
For a humanoid robot walking:
- What's the minimum sensor update frequency needed? (Hz)
- What's the maximum acceptable latency for motor commands? (ms)
- What happens if perception takes too long?

### Challenge 3: Failure Modes
For the coffee cup grasping scenario:
- What could go wrong? (List 5 failure modes)
- Which could be detected with sensors?
- Which require feedback loops to handle?

---

**Ready for the next module?** → [Module 2: Digital Twins & Gazebo](../simulation/intro.md)

You've completed the foundations of Physical AI! You understand why robots are different, how they communicate via ROS 2, and why embodied intelligence requires closed-loop feedback.

Next, we'll learn to **model and simulate** humanoid robots, so you can test your ideas safely before deploying on real hardware.
