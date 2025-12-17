# Capstone Project: Extensions

The capstone project is designed to be a foundation that you can build upon. This guide provides ideas and code templates for extending the system's capabilities.

## How to Modify Each Module

The modular nature of the system makes it easy to swap out or enhance individual components without affecting the rest of the pipeline.

### Modifying Perception

*   **To add a new object for detection**:
    1.  You would typically need to retrain your object detection model (e.g., YOLO) with images of the new object.
    2.  In our mock `camera_processor.py`, you would simply add another `if` condition to create a mock detection for your new object.

### Modifying Planning

*   **To swap the LLM for a different model**:
    1.  Open the `language_planner.py` file.
    2.  Modify the `query_llm_for_plan` function to call your new LLM's API (e.g., switch from an Ollama endpoint to an OpenAI endpoint).
    3.  You may need to adjust your prompt engineering to suit the new model's style.

### Modifying Control

*   **To add a new robot action**:
    1.  **Create a new Action Definition**: In your `interfaces` package, create a new `.action` file (e.g., `Wave.action`).
    2.  **Implement the Action Server**: Create a new Python node (e.g., `wave_controller.py`) that implements an action server for your `Wave` action. This node will contain the logic for the robot's waving motion.
    3.  **Update the Action Dictionary**: In your `language_planner.py`, add "wave" to the list of known actions you send to the LLM in the prompt.
    4.  **Update the Executor**: In `vla_action_executor.py`, create a new action client to call the `/wave` action server.

## Example Extension: Add a New Sensor (Gripper Camera)

**Goal**: Add a camera to the robot's gripper to help with precise object manipulation.

1.  **Update the URDF**: In `humanoid_detailed.urdf`, add a new `<link>` for the gripper camera and a `<joint>` to attach it to the gripper's link. Define a `<sensor>` block for the camera.
2.  **Create a New Perception Node**: Create a `gripper_camera_processor.py` node that subscribes to the new `/gripper_camera/image_raw` topic. This node could perform tasks like detecting if an object is centered in the gripper's view.
3.  **Integrate with Planning**: The LLM could now use this information. For example, it could generate a plan like `[navigate, center_on_object_with_gripper_cam, grasp]`. This requires adding a `center_on_object` action.

## Example Extension: Pick Up Multiple Objects

**Goal**: Handle a command like "Pick up the red ball and the blue cube."

1.  **LLM Planner Update**: The LLM needs to be prompted to recognize multiple objects. Your prompt could instruct it to create a plan with multiple `grasp` actions.
    *   **Plan Output**: `[... grasp('red_ball'), ... grasp('blue_cube')]`
2.  **Control Module Update**: The `grasp` action server might need to be smarter. Does the robot have two hands? Or does it need to place the first object somewhere before grasping the second? The planner and executor must handle this state.
3.  **State Management**: Your `world_state` representation now needs to track which object is in which hand, or where objects have been temporarily placed.

## Code Templates for Common Modifications

### Template: Adding a new Action Server

```python
# new_action_controller.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from interfaces.action import NewAction # Your new .action file

class NewActionController(Node):
    def __init__(self):
        super().__init__('new_action_controller')
        self._action_server = ActionServer(
            self,
            NewAction,
            '/new_action',
            self.execute_callback)
        self.get_logger().info("NewAction server is ready.")

    def execute_callback(self, goal_handle):
        self.get_logger().info("Executing NewAction...")
        # --- Your custom logic here ---
        goal_handle.succeed()
        result = NewAction.Result()
        result.success = True
        return result
```

### Template: Adding a new Perception Node

```python
# new_sensor_processor.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2 # Example for a new sensor type
from std_msgs.msg import Bool

class NewSensorProcessor(Node):
    def __init__(self):
        super().__init__('new_sensor_processor')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/new_sensor/points',
            self.sensor_callback,
            10)
        self.publisher = self.create_publisher(Bool, '/object_is_close', 10)

    def sensor_callback(self, msg):
        # --- Your processing logic here ---
        # e.g., check if any point in the cloud is closer than a threshold
        is_close = self.process_points(msg)
        self.publisher.publish(Bool(data=is_close))
```