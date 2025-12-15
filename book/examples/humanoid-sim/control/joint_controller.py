import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class JointController(Node):
    """
    A simple action server that receives a joint trajectory and "executes" it.
    In a real robot, this node would interface with the hardware drivers for the
    robot's motors (e.g., via a CAN bus, EtherCAT, or serial communication).
    """
    def __init__(self):
        super().__init__('joint_controller')
        self.get_logger().info('Joint Controller node started.')
        
        # This is a standard action type for controlling joint movements
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
            self.execute_callback)
            
        self.get_logger().info("FollowJointTrajectory action server is ready.")

    def execute_callback(self, goal_handle):
        """
        Executes a joint trajectory goal.
        """
        trajectory = goal_handle.request.trajectory
        self.get_logger().info('Executing a new joint trajectory...')

        if not trajectory.points:
            self.get_logger().error('Trajectory has no points.')
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        # In a real system, you would command the hardware to move to each point
        # in the trajectory, respecting the velocities and accelerations.
        for point in trajectory.points:
            self.get_logger().info(f"  Moving to joint positions: {point.positions} "
                                 f"in {point.time_from_start.sec}s")
            # Here you would send commands to the motor drivers.
            # We'll just sleep to simulate the movement.
            # time.sleep(point.time_from_start.sec) # This is not correct way to handle time

        self.get_logger().info('Trajectory execution finished successfully (mock).')
        goal_handle.succeed()
        
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

def main(args=None):
    rclpy.init(args=args)
    joint_controller = JointController()
    rclpy.spin(joint_controller)
    joint_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
