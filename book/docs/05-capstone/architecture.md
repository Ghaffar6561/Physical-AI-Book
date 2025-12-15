# Capstone Project: System Architecture

This document outlines the architecture for the full, end-to-end autonomous humanoid system. The goal of this capstone is to integrate all the concepts from the previous modules into a single, runnable project that can accept spoken commands and perform actions in a simulated environment.

## System Overview

The system is designed as a collection of independent ROS 2 nodes, each responsible for a specific part of the VLA (Vision-Language-Action) pipeline. This modular architecture allows for individual components to be tested, debugged, and upgraded independently.

The high-level data flow is as follows:

1.  A spoken command is captured by the **Voice Interface**.
2.  The command is transcribed to text and sent to the **Planning Module**.
3.  The Planning Module uses an LLM to decompose the command into a sequence of actions.
4.  This action plan is sent to the **Control Module**.
5.  The Control Module executes the actions, using the **Perception Module** to get information about the world.

![Capstone System Diagram](book/static/diagrams/capstone-architecture.svg)  
*(A detailed version of this diagram will be created later)*

## Core Modules (ROS 2 Nodes)

### 1. Perception Module (`/perception`)

*   **Purpose**: To build a representation of the world state from raw sensor data.
*   **Subscriptions**:
    *   `/camera/image_raw`: Raw images from the simulated camera.
    *   `/lidar/scan`: Data from the simulated LiDAR sensor.
    *   `/imu/data`: Inertial data for localization.
*   **Publications**:
    *   `/detected_objects`: A list of objects detected in the scene, with their positions and classifications (e.g., `{name: 'red_ball', position: [x, y, z]}`).
    *   `/occupancy_map`: A 2D map of the environment, showing obstacles.
    *   `/robot_pose`: The estimated position and orientation of the robot in the world.
*   **Implementation**: This module will contain nodes like `camera_processor.py` (for object detection) and `lidar_processor.py` (for mapping).

### 2. Planning Module (`/planning`)

*   **Purpose**: To convert high-level user commands into low-level, executable action plans.
*   **Subscriptions**:
    *   `/user_command`: The text command from the voice interface.
    *   `/world_state`: A topic that aggregates information from the perception module.
*   **Publications**:
    *   `/action_plan`: A sequenced list of actions to be executed (e.g., `[{action: 'navigate', ...}, {action: 'grasp', ...}]`).
*   **Implementation**: This module contains the `language_planner.py` node, which interfaces with the LLM (as detailed in Module 4). It also includes a crucial `action_validator.py` node that sanity-checks the LLM's plan before execution.

### 3. Control Module (`/control`)

*   **Purpose**: To execute the low-level actions from the plan.
*   **Subscriptions**:
    *   `/action_plan`: The plan to be executed.
*   **Action Servers**: This module exposes a set of ROS 2 action servers that represent the robot's physical capabilities.
    *   `/navigate_to_goal`: Action server to move the robot to a specific pose.
    *   `/grasp_object`: Action server to perform a grasping motion.
    *   `/place_object`: Action server to release a grasped object.
*   **Implementation**: This module contains the `motion_planner.py` (which might use a library like MoveIt) and various controller nodes (`joint_controller.py`, `gripper_controller.py`).

### 4. Voice Interface (`/vla`)

*   **Purpose**: To provide a natural language interface for the user.
*   **Publications**:
    *   `/user_command`: The transcribed text of the user's spoken command.
*   **Implementation**: This module contains the `speech_recognizer.py` node, which listens to the microphone, and handles the speech-to-text conversion.

This distributed, message-passing architecture is fundamental to ROS 2 and allows for a scalable and robust robotics system.