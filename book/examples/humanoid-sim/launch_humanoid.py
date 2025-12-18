from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launches all the nodes for the humanoid capstone project.
    """
    return LaunchDescription([
        # --- Perception Module ---
        Node(
            package='humanoid_sim', # Assuming a ROS 2 package 'humanoid_sim'
            executable='camera_processor',
            name='camera_processor'
        ),
        Node(
            package='humanoid_sim',
            executable='lidar_processor',
            name='lidar_processor'
        ),
        Node(
            package='humanoid_sim',
            executable='localization',
            name='localization_node'
        ),
        
        # --- VLA Module ---
        Node(
            package='humanoid_sim',
            executable='speech_recognizer',
            name='speech_recognizer'
        ),
        Node(
            package='humanoid_sim',
            executable='language_planner',
            name='language_planner'
        ),
        Node(
            package='humanoid_sim',
            executable='action_executor',
            name='vla_action_executor'
        ),
        
        # --- Planning Module ---
        Node(
            package='humanoid_sim',
            executable='task_planner',
            name='task_planner'
        ),
        Node(
            package='humanoid_sim',
            executable='action_validator',
            name='action_validator'
        ),
        Node(
            package='humanoid_sim',
            executable='motion_planner',
            name='motion_planner'
        ),
        
        # --- Control Module ---
        Node(
            package='humanoid_sim',
            executable='joint_controller',
            name='joint_controller'
        ),
        Node(
            package='humanoid_sim',
            executable='gripper_controller',
            name='gripper_controller'
        ),
        Node(
            package='humanoid_sim',
            executable='locomotion_controller',
            name='locomotion_controller'
        ),

        # In a real launch file, you would also include the Gazebo launch
        # and the robot_state_publisher for the URDF.
        # e.g., IncludeLaunchDescription(
        #          PythonLaunchDescriptionSource([os.path.join(
        #              get_package_share_directory('gazebo_ros'), 'launch'), '/gazebo.launch.py'])),
    ])
