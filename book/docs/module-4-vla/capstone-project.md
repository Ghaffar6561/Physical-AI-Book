---
sidebar_position: 10
title: "Capstone Project"
---

# Capstone Project: The Autonomous Humanoid

## Overview

The capstone project integrates everything learned across all four modules into a complete autonomous humanoid system. The robot will:

1. **Receive a voice command** (Whisper)
2. **Plan a sequence of actions** (GPT/LLM)
3. **Navigate to a location** (Nav2 + VSLAM)
4. **Identify an object** (Computer Vision)
5. **Manipulate the object** (Inverse Kinematics + Control)

## Project Specification

```
┌─────────────────────────────────────────────────────────────┐
│               Capstone System Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  "Pick up the red cup from the kitchen table"               │
│                          │                                   │
│                          ▼                                   │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│   │   Whisper   │──▶│    GPT-4    │──▶│   Task      │       │
│   │     ASR     │   │   Planner   │   │   Queue     │       │
│   └─────────────┘   └─────────────┘   └──────┬──────┘       │
│                                              │               │
│   ┌──────────────────────────────────────────▼──────────┐   │
│   │                    Task Executor                     │   │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │   │
│   │  │Navigate │─▶│ Detect  │─▶│  Grasp  │─▶│ Return │ │   │
│   │  │to kitchen│ │red cup  │  │  cup    │  │to user │ │   │
│   │  └─────────┘  └─────────┘  └─────────┘  └────────┘ │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   Robot Hardware                     │   │
│   │  • ROS 2 Humble  • Isaac ROS  • Jetson Orin        │   │
│   │  • RealSense D435i  • Microphone Array             │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## System Requirements

### Hardware (Simulation)
- RTX 4070+ GPU with 12GB+ VRAM
- 64GB RAM
- Ubuntu 22.04

### Hardware (Physical Deployment)
- Jetson Orin NX/AGX
- Intel RealSense D435i
- USB Microphone Array
- Humanoid robot (simulated or Unitree G1)

### Software
- ROS 2 Humble
- Isaac Sim 2023.1.1+
- Isaac ROS 2.0+
- Whisper (base model)
- OpenAI API (GPT-4)

## Project Structure

```
humanoid_capstone/
├── humanoid_capstone/
│   ├── __init__.py
│   ├── voice_interface.py      # Whisper + wake word
│   ├── task_planner.py         # GPT task planning
│   ├── navigation.py           # Nav2 integration
│   ├── perception.py           # Object detection
│   ├── manipulation.py         # Grasping control
│   └── main_controller.py      # Orchestrator
├── launch/
│   ├── simulation.launch.py    # Isaac Sim launch
│   ├── perception.launch.py    # Camera + detection
│   └── full_system.launch.py   # Complete system
├── config/
│   ├── nav2_params.yaml
│   ├── perception_params.yaml
│   └── robot_config.yaml
├── models/
│   ├── whisper/                # ASR model
│   └── yolo/                   # Object detection
├── worlds/
│   └── apartment.usd           # Isaac Sim world
└── urdf/
    └── humanoid.urdf           # Robot description
```

## Implementation Guide

### Step 1: Voice Interface

```python
# voice_interface.py
from faster_whisper import WhisperModel
import pvporcupine
import pyaudio
import numpy as np

class VoiceInterface:
    def __init__(self):
        # Wake word detector
        self.porcupine = pvporcupine.create(
            keywords=["jarvis"]  # Or custom wake word
        )

        # Whisper ASR
        self.whisper = WhisperModel("base", device="cuda")

        # Audio setup
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            rate=16000,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=512
        )

    def listen_for_command(self) -> str:
        """Listen for wake word, then transcribe command."""
        print("Listening for wake word...")

        while True:
            pcm = self.stream.read(512)
            pcm = np.frombuffer(pcm, dtype=np.int16)

            if self.porcupine.process(pcm) >= 0:
                print("Wake word detected! Listening...")
                return self.capture_and_transcribe()

    def capture_and_transcribe(self) -> str:
        """Capture audio until silence, then transcribe."""
        frames = []
        silence_count = 0

        while silence_count < 30:  # ~1.5 seconds of silence
            data = self.stream.read(512)
            frames.append(data)

            # Check for silence
            audio = np.frombuffer(data, dtype=np.int16)
            if np.abs(audio).mean() < 500:
                silence_count += 1
            else:
                silence_count = 0

        # Transcribe
        audio = np.concatenate([
            np.frombuffer(f, dtype=np.int16) for f in frames
        ]).astype(np.float32) / 32768.0

        segments, _ = self.whisper.transcribe(audio)
        return " ".join([s.text for s in segments]).strip()
```

### Step 2: Task Planner

```python
# task_planner.py
import openai
import json

PLANNER_PROMPT = """
You are a task planner for a humanoid robot. Convert natural language commands
into a sequence of executable actions.

Available actions:
- navigate(location): Move to a named location
- scan_for_object(object_name): Look for an object
- approach_object(object_id): Move close to detected object
- grasp(object_id): Pick up the object
- release(): Put down held object
- speak(message): Say something
- wait(seconds): Pause execution

Output a JSON array of actions:
[{"action": "name", "params": {...}}, ...]

Consider:
1. Always scan before approaching
2. Announce what you're doing
3. Handle potential failures
"""

class TaskPlanner:
    def __init__(self):
        self.client = openai.OpenAI()

    def plan(self, command: str) -> list:
        """Convert natural language to action sequence."""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": command}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("actions", [])
```

### Step 3: Navigation

```python
# navigation.py
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped

class NavigationController:
    def __init__(self, node):
        self.navigator = BasicNavigator()

        # Predefined locations
        self.locations = {
            "kitchen": self.create_pose(5.0, 2.0, 0.0),
            "living_room": self.create_pose(0.0, 0.0, 0.0),
            "bedroom": self.create_pose(-3.0, 4.0, 1.57),
            "entrance": self.create_pose(0.0, -5.0, 0.0),
        }

    def create_pose(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = np.sin(yaw / 2)
        pose.pose.orientation.w = np.cos(yaw / 2)
        return pose

    def navigate_to(self, location: str) -> bool:
        """Navigate to named location."""
        if location not in self.locations:
            return False

        goal = self.locations[location]
        self.navigator.goToPose(goal)

        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            # Could publish progress here

        result = self.navigator.getResult()
        return result == TaskResult.SUCCEEDED
```

### Step 4: Perception

```python
# perception.py
from ultralytics import YOLO
import cv2
import numpy as np

class PerceptionSystem:
    def __init__(self):
        self.detector = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        # Object tracking
        self.detected_objects = {}

    def process_image(self, image_msg) -> list:
        """Detect objects in image."""
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')

        results = self.detector(cv_image)

        objects = []
        for r in results:
            for box in r.boxes:
                obj = {
                    'id': len(objects),
                    'class': self.detector.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist(),
                    'center': self.get_center(box.xyxy[0])
                }
                objects.append(obj)

        return objects

    def find_object(self, target_name: str, color: str = None) -> dict:
        """Find specific object by name and optional color."""
        for obj in self.detected_objects.values():
            if target_name.lower() in obj['class'].lower():
                if color is None:
                    return obj
                # Check color in ROI
                if self.check_color(obj, color):
                    return obj
        return None
```

### Step 5: Manipulation

```python
# manipulation.py
class ManipulationController:
    def __init__(self, node):
        self.node = node

        # Inverse kinematics solver
        self.ik_solver = IKSolver('/urdf/humanoid.urdf')

        # Joint command publisher
        self.joint_pub = node.create_publisher(
            JointState, '/humanoid/joint_commands', 10
        )

    def grasp_object(self, object_pose) -> bool:
        """Execute grasp motion for object at given pose."""
        # Pre-grasp pose (10cm above object)
        pre_grasp = object_pose.copy()
        pre_grasp[2] += 0.10

        # Move to pre-grasp
        if not self.move_hand_to(pre_grasp):
            return False

        # Open gripper
        self.set_gripper(1.0)

        # Move to grasp pose
        if not self.move_hand_to(object_pose):
            return False

        # Close gripper
        self.set_gripper(0.0)

        # Lift
        lift_pose = object_pose.copy()
        lift_pose[2] += 0.15
        self.move_hand_to(lift_pose)

        return True

    def move_hand_to(self, target_position) -> bool:
        """Move hand to target using IK."""
        joint_angles = self.ik_solver.solve(
            target_position,
            end_effector='right_hand'
        )

        if joint_angles is None:
            return False

        msg = JointState()
        msg.name = list(joint_angles.keys())
        msg.position = list(joint_angles.values())
        self.joint_pub.publish(msg)

        return True
```

### Step 6: Main Controller

```python
# main_controller.py
class CapstoneController(Node):
    def __init__(self):
        super().__init__('capstone_controller')

        # Initialize subsystems
        self.voice = VoiceInterface()
        self.planner = TaskPlanner()
        self.navigation = NavigationController(self)
        self.perception = PerceptionSystem()
        self.manipulation = ManipulationController(self)

        # TTS for feedback
        self.tts_pub = self.create_publisher(String, '/robot/speak', 10)

    def run(self):
        """Main control loop."""
        while rclpy.ok():
            # Wait for voice command
            command = self.voice.listen_for_command()
            self.speak(f"I heard: {command}")

            # Plan actions
            actions = self.planner.plan(command)
            self.speak(f"I'll execute {len(actions)} actions")

            # Execute each action
            for action in actions:
                success = self.execute_action(action)
                if not success:
                    self.speak("I encountered a problem. Stopping.")
                    break

            self.speak("Task complete!")

    def execute_action(self, action) -> bool:
        """Execute a single action."""
        action_type = action['action']
        params = action.get('params', {})

        if action_type == 'navigate':
            return self.navigation.navigate_to(params['location'])

        elif action_type == 'scan_for_object':
            obj = self.perception.find_object(params['object_name'])
            return obj is not None

        elif action_type == 'grasp':
            obj = self.perception.get_object(params['object_id'])
            if obj:
                pose = self.perception.get_3d_position(obj)
                return self.manipulation.grasp_object(pose)
            return False

        elif action_type == 'speak':
            self.speak(params['message'])
            return True

        return False

    def speak(self, text):
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)
```

## Testing & Evaluation

### Test Scenarios

| Scenario | Command | Expected Actions |
|----------|---------|-----------------|
| Simple fetch | "Get me water" | Navigate → Detect → Grasp → Return |
| Color-specific | "Bring the red apple" | Navigate → Detect red → Grasp → Return |
| Multi-step | "Clean the table" | Navigate → Detect items → Grasp each → Place in bin |
| Error handling | "Get the elephant" | Navigate → Fail to detect → Report failure |

### Evaluation Metrics

- **Task Success Rate**: % of commands completed successfully
- **Response Time**: Voice command to first action
- **Navigation Accuracy**: Distance from goal position
- **Grasp Success Rate**: % of successful object pickups
- **User Satisfaction**: Subjective rating of interaction

## Demo Script

```bash
# Terminal 1: Launch Isaac Sim
ros2 launch humanoid_capstone simulation.launch.py

# Terminal 2: Launch perception
ros2 launch humanoid_capstone perception.launch.py

# Terminal 3: Launch main controller
ros2 run humanoid_capstone main_controller

# Say: "Hey robot, get me the red cup from the kitchen"
```

## Extensions

1. **Multi-object manipulation**: Handle multiple objects in sequence
2. **Dynamic replanning**: Adapt when obstacles appear
3. **Learning from demonstration**: Record and replay human demonstrations
4. **Multi-robot coordination**: Multiple humanoids working together
5. **Real robot deployment**: Transfer to Unitree G1

---

*Congratulations! You've built a complete autonomous humanoid system integrating perception, planning, navigation, and manipulation.*
