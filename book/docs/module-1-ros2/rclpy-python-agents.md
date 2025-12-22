---
sidebar_position: 4
title: "Python Agents with rclpy"
---

# Bridging Python AI Agents to ROS 2 Controllers

## Overview

One of ROS 2's greatest strengths is its native Python support through **rclpy**. This enables direct integration between AI/ML frameworks (PyTorch, TensorFlow, OpenAI APIs) and robot controllers. You can run neural networks, language models, and vision systems as ROS 2 nodes that communicate seamlessly with the robot hardware.

## The AI-to-Robot Bridge Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 AI Agent ←→ ROS 2 Bridge                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐     rclpy      ┌─────────────────┐       │
│   │   PyTorch   │◀──────────────▶│  ROS 2 Topics   │       │
│   │   Model     │                │  /camera/image  │       │
│   └─────────────┘                └─────────────────┘       │
│                                                             │
│   ┌─────────────┐     rclpy      ┌─────────────────┐       │
│   │   OpenAI    │◀──────────────▶│  ROS 2 Actions  │       │
│   │   GPT API   │                │  /execute_task  │       │
│   └─────────────┘                └─────────────────┘       │
│                                                             │
│   ┌─────────────┐     rclpy      ┌─────────────────┐       │
│   │   Vision    │◀──────────────▶│  ROS 2 Services │       │
│   │   Model     │                │  /detect_object │       │
│   └─────────────┘                └─────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Creating an AI Agent Node

### Basic Structure

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import numpy as np

class VisionAIAgent(Node):
    """
    AI Agent that processes camera images and controls robot movement.
    """
    def __init__(self):
        super().__init__('vision_ai_agent')

        # Load AI model
        self.model = self.load_model()
        self.bridge = CvBridge()

        # Subscribe to camera
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publish velocity commands
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.get_logger().info('Vision AI Agent initialized')

    def load_model(self):
        """Load PyTorch model for inference."""
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model.eval()
        return model

    def image_callback(self, msg):
        """Process incoming images with AI model."""
        # Convert ROS Image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Run inference
        results = self.model(cv_image)
        detections = results.pandas().xyxy[0]

        # Generate control command based on detections
        cmd = self.generate_command(detections)
        self.cmd_pub.publish(cmd)

    def generate_command(self, detections):
        """Convert AI detections to robot commands."""
        cmd = Twist()

        # Example: Follow detected person
        persons = detections[detections['name'] == 'person']
        if len(persons) > 0:
            # Calculate center of first person
            person = persons.iloc[0]
            center_x = (person['xmin'] + person['xmax']) / 2

            # Turn towards person (assuming 640px width)
            error = (center_x - 320) / 320
            cmd.angular.z = -error * 0.5  # Proportional control

            # Move forward if person detected
            cmd.linear.x = 0.2

        return cmd
```

## Integrating Large Language Models

### LLM-Powered Task Planning

```python
import openai
from std_msgs.msg import String
from humanoid_interfaces.action import ExecuteTask

class LLMPlannerAgent(Node):
    """
    Uses GPT to convert natural language commands to robot actions.
    """
    def __init__(self):
        super().__init__('llm_planner')

        # OpenAI client
        self.client = openai.OpenAI()

        # Subscribe to voice commands
        self.voice_sub = self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        # Action client for task execution
        self.task_client = ActionClient(
            self,
            ExecuteTask,
            '/execute_task'
        )

        # System prompt for robot planning
        self.system_prompt = """
        You are a humanoid robot planner. Convert natural language commands
        into a sequence of robot actions.

        Available actions:
        - walk_to(location): Navigate to a location
        - pick_up(object): Grasp an object
        - place(object, location): Put object somewhere
        - look_at(target): Turn head towards target
        - say(message): Speak to human

        Output JSON format:
        {"actions": [{"type": "action_name", "params": {...}}]}
        """

    def command_callback(self, msg):
        """Process natural language command."""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Get action plan from LLM
        plan = self.get_action_plan(command)

        # Execute each action
        for action in plan['actions']:
            self.execute_action(action)

    def get_action_plan(self, command):
        """Query LLM for action sequence."""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": command}
            ],
            response_format={"type": "json_object"}
        )

        import json
        return json.loads(response.choices[0].message.content)

    def execute_action(self, action):
        """Send action to robot controller."""
        goal = ExecuteTask.Goal()
        goal.action_type = action['type']
        goal.parameters = str(action['params'])

        self.task_client.send_goal_async(goal)
```

## Real-Time Inference Considerations

### Threading for Non-Blocking AI

```python
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading

class RealTimeAIAgent(Node):
    def __init__(self):
        super().__init__('realtime_ai_agent')

        # Use reentrant callback group for parallel processing
        self.cb_group = ReentrantCallbackGroup()

        # High-frequency sensor subscription
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10,
            callback_group=self.cb_group
        )

        # Separate thread for heavy AI inference
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.start()

        self.latest_image = None
        self.image_lock = threading.Lock()

    def inference_loop(self):
        """Run heavy inference in separate thread."""
        while rclpy.ok():
            with self.image_lock:
                if self.latest_image is not None:
                    # Heavy AI processing here
                    result = self.model.predict(self.latest_image)
                    self.process_result(result)
            time.sleep(0.033)  # ~30 Hz

def main():
    rclpy.init()
    node = RealTimeAIAgent()

    # Use multi-threaded executor for parallel callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
```

## Message Type Conversions

### Common Conversions for AI

```python
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class DataConverter:
    def __init__(self):
        self.bridge = CvBridge()

    def image_to_numpy(self, ros_image):
        """ROS Image → NumPy array for PyTorch/TensorFlow."""
        return self.bridge.imgmsg_to_cv2(ros_image, 'rgb8')

    def numpy_to_image(self, np_array):
        """NumPy array → ROS Image."""
        return self.bridge.cv2_to_imgmsg(np_array, 'rgb8')

    def pointcloud_to_numpy(self, pc_msg):
        """PointCloud2 → NumPy array for 3D processing."""
        points = []
        for point in pc2.read_points(pc_msg, field_names=['x', 'y', 'z']):
            points.append([point[0], point[1], point[2]])
        return np.array(points)

    def numpy_to_tensor(self, np_array):
        """NumPy → PyTorch tensor (GPU if available)."""
        import torch
        tensor = torch.from_numpy(np_array).float()
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor
```

## Package Structure for AI Agents

```
humanoid_ai_agents/
├── humanoid_ai_agents/
│   ├── __init__.py
│   ├── vision_agent.py
│   ├── llm_planner.py
│   ├── vla_controller.py
│   └── models/
│       ├── __init__.py
│       └── load_models.py
├── launch/
│   └── ai_agents.launch.py
├── config/
│   ├── model_config.yaml
│   └── agent_params.yaml
├── resource/
│   └── humanoid_ai_agents
├── package.xml
├── setup.py
└── setup.cfg
```

### setup.py for AI Package

```python
from setuptools import setup

package_name = 'humanoid_ai_agents'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=[
        'setuptools',
        'torch',
        'opencv-python',
        'openai',
    ],
    entry_points={
        'console_scripts': [
            'vision_agent = humanoid_ai_agents.vision_agent:main',
            'llm_planner = humanoid_ai_agents.llm_planner:main',
            'vla_controller = humanoid_ai_agents.vla_controller:main',
        ],
    },
)
```

## Key Takeaways

1. **rclpy** enables seamless Python AI integration with ROS 2
2. Use **cv_bridge** for image conversions between ROS and OpenCV/NumPy
3. Run heavy inference in **separate threads** to avoid blocking
4. Use **MultiThreadedExecutor** for parallel callback processing
5. Structure AI agents as proper ROS 2 packages for reusability

---

*Next: Learn about launch files and parameter management for deploying AI agent systems.*
