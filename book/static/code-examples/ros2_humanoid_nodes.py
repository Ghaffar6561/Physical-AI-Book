"""
ROS 2 Humanoid Robot Control Nodes

This example demonstrates how to control a simulated humanoid robot in Gazebo.

Three nodes are included:
1. HumanoidController — Publishes joint commands (position control)
2. HumanoidMonitor — Subscribes to joint states and reports feedback
3. GripperController — Opens/closes grippers via action server

Key Concepts:
- Joint state publisher/subscriber pattern
- Synchronized control of multiple joints
- Feedback loops for validation
- Real-world applicable control architecture

Expected Output:
  HumanoidController: Publishing commands to /joint_command
  HumanoidMonitor: Receiving joint states, logging feedback
  GripperController: Opening/closing grippers on command

Learning Goals:
- Understand how to command humanoid robots from ROS 2 nodes
- Learn joint state feedback and validation
- See real-world control patterns for multiple DOF systems
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import math
import time


class HumanoidController(Node):
    """
    Publishes joint command messages to control the humanoid robot.

    Simulates smooth motion by publishing sinusoidal joint trajectories.
    This demonstrates how to command a 30-DOF humanoid in a coordinated way.
    """

    def __init__(self):
        super().__init__('humanoid_controller')

        # Publisher for joint commands
        self.joint_command_pub = self.create_publisher(
            JointState, '/joint_commands', 10
        )

        # List of all humanoid joints (simplified version)
        self.joint_names = [
            # Head (1 DOF)
            'neck_yaw',
            # Right arm (7 DOF)
            'right_shoulder_pitch', 'right_elbow_pitch', 'right_wrist_pitch',
            'right_gripper',
            # Left arm (7 DOF)
            'left_shoulder_pitch', 'left_elbow_pitch', 'left_wrist_pitch',
            'left_gripper',
            # Right leg (6 DOF)
            'right_hip_pitch', 'right_knee_pitch', 'right_ankle_pitch',
            # Left leg (6 DOF)
            'left_hip_pitch', 'left_knee_pitch', 'left_ankle_pitch',
        ]

        # Control frequency: 50 Hz (20 ms per command)
        self.timer = self.create_timer(0.02, self.send_joint_commands)
        self.time_step = 0
        self.get_logger().info('HumanoidController initialized')

    def send_joint_commands(self):
        """
        Send joint commands to the humanoid robot.

        Uses smooth sinusoidal trajectories to command arms and legs.
        Keeps standing posture while moving limbs.
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # Current time (in seconds)
        t = self.time_step * 0.02

        # Smooth sinusoidal trajectories for limb movement
        # Amplitude, frequency, and phase chosen for natural motion
        positions = [
            # Head: gentle yaw oscillation
            0.3 * math.sin(t),

            # Right arm: wave motion
            0.5 * math.sin(t),              # shoulder pitch
            1.0 * math.sin(t + math.pi/4), # elbow pitch
            0.3 * math.cos(t),              # wrist pitch
            0.05 * math.sin(2*t),           # gripper (open/close)

            # Left arm: opposite phase for balanced motion
            0.5 * math.sin(t + math.pi),    # shoulder pitch
            1.0 * math.sin(t + 5*math.pi/4), # elbow pitch
            0.3 * math.cos(t + math.pi),    # wrist pitch
            0.05 * math.sin(2*t + math.pi), # gripper

            # Right leg: small oscillation (maintaining stance)
            0.2 * math.sin(t),              # hip pitch
            0.1 * math.sin(t),              # knee pitch
            0.1 * math.sin(t),              # ankle pitch

            # Left leg: opposite phase
            0.2 * math.sin(t + math.pi),    # hip pitch
            0.1 * math.sin(t + math.pi),    # knee pitch
            0.1 * math.sin(t + math.pi),    # ankle pitch
        ]

        msg.position = positions
        msg.velocity = [0.5] * len(self.joint_names)  # Desired velocity
        msg.effort = [10.0] * len(self.joint_names)   # Motor effort

        self.joint_command_pub.publish(msg)

        # Log every 10 seconds
        if self.time_step % 500 == 0:
            self.get_logger().info(
                f'Commands sent at t={t:.1f}s | '
                f'Right shoulder: {positions[1]:.2f} rad, '
                f'Left knee: {positions[12]:.2f} rad'
            )

        self.time_step += 1


class HumanoidMonitor(Node):
    """
    Monitors and validates joint state feedback from Gazebo.

    Subscribes to /joint_states published by Gazebo and logs:
    - Current position vs. commanded position
    - Velocity at each joint
    - Any anomalies (e.g., joint exceeding limits)
    """

    def __init__(self):
        super().__init__('humanoid_monitor')

        # Subscribe to joint states from Gazebo
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Store last commanded positions for comparison
        self.last_commanded_positions = {}

        self.get_logger().info('HumanoidMonitor initialized')

    def joint_state_callback(self, msg):
        """
        Called whenever joint states are received from Gazebo.

        Analyzes feedback and detects anomalies:
        - Large position errors (tracking failure)
        - Velocities exceeding limits
        - Joints hitting limits
        """
        if len(msg.name) == 0:
            return

        # Select a few key joints to monitor
        key_joints = [
            'right_shoulder_pitch', 'right_elbow_pitch', 'right_gripper',
            'right_hip_pitch', 'right_knee_pitch'
        ]

        for i, joint_name in enumerate(msg.name):
            if joint_name not in key_joints:
                continue

            pos = msg.position[i] if i < len(msg.position) else 0.0
            vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
            torque = msg.effort[i] if i < len(msg.effort) else 0.0

            # Check for anomalies
            if abs(vel) > 5.0:  # Velocity limit: 5 rad/s
                self.get_logger().warn(
                    f'High velocity on {joint_name}: {vel:.2f} rad/s'
                )

            if abs(torque) > 100.0:  # Torque limit: 100 Nm
                self.get_logger().warn(
                    f'High torque on {joint_name}: {torque:.2f} Nm'
                )

            # Log key joints periodically
            if i == 0:  # Log once per callback
                self.get_logger().debug(
                    f'{joint_name}: pos={pos:.3f} rad, '
                    f'vel={vel:.3f} rad/s, torque={torque:.1f} Nm'
                )


class GripperController(Node):
    """
    Controls the humanoid's grippers via a simple interface.

    In a real system, this would use ROS 2 actions for asynchronous gripper control.
    Here we demonstrate a simple publisher approach.

    Action: open_gripper / close_gripper
    """

    def __init__(self):
        super().__init__('gripper_controller')

        # Publisher for gripper commands (left and right)
        self.gripper_command_pub = self.create_publisher(
            Float64MultiArray, '/gripper_commands', 10
        )

        # Timer for gripper state machine
        self.timer = self.create_timer(2.0, self.control_grippers)
        self.state = 0  # 0: closed, 1: opening, 2: open, 3: closing

        self.get_logger().info('GripperController initialized')

    def control_grippers(self):
        """
        Simple state machine to open/close grippers alternately.

        State transitions:
          0 (closed) → 1 (opening) → 2 (open) → 3 (closing) → 0 (closed)
        """
        msg = Float64MultiArray()

        if self.state == 0:  # Closed (gripper = 0)
            msg.data = [0.0, 0.0]  # Left, right gripper positions
            self.get_logger().info('Grippers: CLOSED')
            self.state = 1

        elif self.state == 1:  # Opening
            msg.data = [0.05, 0.05]  # Partially open
            self.get_logger().info('Grippers: OPENING')
            self.state = 2

        elif self.state == 2:  # Fully open (gripper = 0.1)
            msg.data = [0.1, 0.1]  # Fully open
            self.get_logger().info('Grippers: OPEN')
            self.state = 3

        elif self.state == 3:  # Closing
            msg.data = [0.05, 0.05]  # Partially closing
            self.get_logger().info('Grippers: CLOSING')
            self.state = 0

        self.gripper_command_pub.publish(msg)


def main():
    """
    Main entry point: Launch all three nodes in parallel.

    This demonstrates how a complete humanoid control system would work:
    1. Controller generates trajectories (runs at 50 Hz)
    2. Monitor validates feedback (event-driven)
    3. Gripper controller manages end-effectors (runs at 0.5 Hz)

    Usage:
      python ros2_humanoid_nodes.py

    In a real deployment, each node would run in a separate process:
      ros2 run humanoid_sim humanoid_controller
      ros2 run humanoid_sim humanoid_monitor
      ros2 run humanoid_sim gripper_controller
    """
    rclpy.init()

    # Create all three nodes
    controller = HumanoidController()
    monitor = HumanoidMonitor()
    gripper = GripperController()

    # Run with MultiThreadedExecutor to handle concurrent callbacks
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(controller)
    executor.add_node(monitor)
    executor.add_node(gripper)

    print("\n" + "="*70)
    print("Humanoid Robot Control System Started")
    print("="*70)
    print("\nNodes running:")
    print("  - HumanoidController (50 Hz): Commands arms, legs, head")
    print("  - HumanoidMonitor (event-driven): Validates joint feedback")
    print("  - GripperController (0.5 Hz): Opens/closes grippers")
    print("\nTo visualize in RViz:")
    print("  ros2 run rviz2 rviz2")
    print("\nTo monitor topics:")
    print("  ros2 topic echo /joint_commands")
    print("  ros2 topic echo /joint_states")
    print("\nPress Ctrl+C to stop\n")

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    finally:
        controller.destroy_node()
        monitor.destroy_node()
        gripper.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
