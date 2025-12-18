# Capstone Project: Running the System

This guide explains how to launch and interact with the complete humanoid robot capstone system in the Gazebo simulation environment.

## Prerequisites

*   You must have successfully completed all the steps in the [Setup Guide](./setup.md).
*   Ensure you have sourced your ROS 2 workspace in your current terminal:
    ```bash
    source install/setup.bash
    ```

## 1. Launching the System

The entire system, including the Gazebo simulation and all the ROS 2 nodes, can be brought up with a single launch file.

```bash
# From the root of your ros2_ws
ros2 launch humanoid_sim launch_humanoid.py
```

This command will:
1.  Start the Gazebo simulator and load the `simple_world.sdf`.
2.  Spawn the detailed humanoid URDF model in Gazebo.
3.  Launch all the ROS 2 nodes from the perception, planning, control, and VLA modules.

You should see a flood of log messages from all the starting nodes, and two windows should appear: one for the Gazebo client (showing the simulated world) and one for RViz (visualizing the robot's state).

## 2. How to Issue Spoken Commands

The `speech_recognizer` node is designed to listen for commands periodically.

1.  **Wait for Calibration**: When you first launch the system, the speech recognizer will take a moment to calibrate for ambient noise. You will see a log message like `Calibration complete. Ready for commands.`
2.  **Speak Clearly**: Once it's ready, simply speak a command into your microphone. For example:
    > "Hey robot, get me the red ball from the table."
3.  **Monitor the Logs**: You can watch the terminal where you launched the system to see the pipeline in action:
    *   The `speech_recognizer` will log the transcribed text.
    *   The `language_planner` will log the command it's planning for.
    *   The `action_executor` will log the steps of the plan it's executing.

## 3. Expected Output and Examples

### Example 1: Simple Navigation

*   **Command**: "Robot, go to the table."
*   **Expected Behavior**:
    1.  The `language_planner` creates a plan: `[navigate('table')]`.
    2.  The `action_executor` calls the `/navigate_to_goal` action server.
    3.  The `locomotion_controller` drives the robot toward the table in Gazebo.

### Example 2: Full Pick-and-Place

*   **Command**: "Bring me the coke can."
*   **Expected Behavior**:
    1.  The LLM creates a multi-step plan: `[navigate('table'), grasp('coke_can'), navigate('user'), place('user')]`.
    2.  The robot first moves to the table.
    3.  The robot's arm moves to grasp the can (you would see this in Gazebo).
    4.  The robot returns to its starting position (simulating 'user' location).
    5.  The robot's arm moves to a placing position.

## 4. Debugging Tips

If the system isn't behaving as you expect, here are some useful ROS 2 commands for debugging:

*   **Check Active Nodes**: See if all your nodes are running.
    ```bash
    ros2 node list
    ```
*   **Check Active Topics**: See what topics are being published to.
    ```bash
    ros2 topic list
    ```
*   **Echo a Topic**: View the messages being published on a specific topic. This is incredibly useful for seeing the output of a specific module.
    ```bash
    # See the output of the speech recognizer
    ros2 topic echo /user_command
    
    # See the validated action plan
    ros2 topic echo /validated_action_plan
    ```
*   **View the TF Tree**: Check the transform tree to debug localization issues.
    ```bash
    ros2 run tf2_tools view_frames
    ```
*   **Use RViz**: RViz is a powerful visualization tool. You can add displays for camera images, LiDAR scans, occupancy maps, and robot poses to get a clear picture of what the robot is "thinking".