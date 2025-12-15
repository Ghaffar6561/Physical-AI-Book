# Autonomous Humanoid Robot Capstone Project

This directory contains the complete capstone project: an autonomous humanoid robot that executes spoken commands in Gazebo simulation.

## Overview

The capstone project integrates all modules from the Physical AI book into a working system:

1. **Perception Module**: Processes camera and LiDAR data to detect objects and navigate
2. **Planning Module**: Uses LLM to decompose spoken commands into executable actions
3. **Control Module**: Executes joint commands for locomotion and manipulation
4. **Voice Interface**: Converts speech to text and sends to LLM for processing

## Architecture

```
Spoken Command → Whisper/speech-recognition → Text
Text → LLM (Llama 2/Mistral) → Semantic Action Plan
Action Plan → ROS 2 Action Calls → Robot Execution
Sensor Feedback → LLM Context → Adaptive Replanning
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   cd book/examples/humanoid-sim
   python -m pip install -r ../requirements.txt
   ```

2. **Launch the System**:
   ```bash
   python launch_humanoid.py
   ```

3. **Issue Commands**:
   The system will prompt for voice commands. Try:
   - "Go to the red ball"
   - "Pick up the object and place it on the table"
   - "Fetch the item from the shelf"

## File Structure

- `ros2_nodes/` — Core ROS 2 node implementations
  - `perception.py` — Processes camera/LiDAR data
  - `planning.py` — Task decomposition and action planning
  - `control.py` — Joint control and trajectory execution
  - `voice_interface.py` — Speech recognition and LLM integration

- `gazebo_models/` — URDF and SDF models for simulation
  - `humanoid_simple.urdf` — Basic humanoid model (10+ joints)
  - `humanoid_detailed.urdf` — More sophisticated humanoid model
  - `simple_world.sdf` — Gazebo environment (table, objects)
  - `chair/` — Chair model for manipulation
  - `door/` — Door model with hinges
  - `shelf/` — Shelf model with multiple levels
  - `gripper/` — Gripper model for manipulation

- `perception/` — Computer vision and SLAM algorithms
  - `camera_processor.py` — Object detection from camera
  - `lidar_processor.py` — Occupancy mapping from LiDAR
  - `localization.py` — Robot localization

- `planning/` — Task and motion planning algorithms
  - `task_planner.py` — LLM-based task decomposition
  - `action_validator.py` — Validate actions for safety
  - `motion_planner.py` — Path planning

- `control/` — Robot control algorithms
  - `joint_controller.py` — Joint command execution
  - `gripper_controller.py` — Gripper control
  - `locomotion_controller.py` — Walking/navigation

- `vla/` — Vision-language-action components
  - `speech_recognizer.py` — Speech-to-text processing
  - `language_planner.py` — LLM inference for action planning
  - `action_executor.py` — Executes parsed actions

## API Documentation

### ROS 2 Topics
- `/camera/image_raw` — Camera image data
- `/lidar/scan` — LiDAR point cloud data
- `/imu/data` — IMU sensor data
- `/joint_states` — Current joint positions

### ROS 2 Actions
- `/navigate_to_pose` — Navigate to a specific pose
- `/pick_object` — Pick up an object
- `/place_object` — Place an object at a location
- `/open_door` — Open a door

### ROS 2 Services
- `/get_object_location` — Get location of specified object
- `/validate_action` — Validate an action before execution

## Requirements

- Python 3.9+
- ROS 2 Humble or Jazzy
- Gazebo 11+
- OpenCV
- Speech recognition libraries
- LLM (Llama 2/Mistral or OpenAI GPT-4)

## Links to Book Chapters

For detailed explanations of each component, refer to these book chapters:

- **Module 1: Physical AI Foundations** — Core concepts behind embodied intelligence
- **Module 2: Digital Twins & Gazebo** — Simulation environment setup
- **Module 3: Perception & NVIDIA Isaac** — Perception algorithms and sensor fusion
- **Module 4: Vision-Language-Action Systems** — Voice interface and LLM integration
- **Module 5: Capstone Project** — Complete system integration

## Troubleshooting

- **Gazebo not starting**: Ensure Gazebo 11+ is installed and in your PATH
- **LLM not responding**: Check internet connection and API key configuration
- **Robot not moving**: Verify ROS 2 nodes are communicating correctly
- **No speech recognition**: Ensure microphone permissions are granted

## Extending the Capstone

The system is designed to be modular and extensible:

1. **New Actions**: Add action servers in `control/` and update `action_executor.py`
2. **New Sensors**: Add sensor processing in `perception/` and update perception nodes
3. **Different LLM**: Modify `language_planner.py` to use a different LLM
4. **New Environments**: Add SDF world files in `gazebo_models/`

For more information on extending the system, see `extensions.md` in the book documentation.