import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from control_msgs.action import GripperCommand

class GripperController(Node):
    """
    An action server for controlling the state of the robot's gripper.
    This is a common way to abstract gripper control in robotics.
    """
    def __init__(self):
        super().__init__('gripper_controller')
        self.get_logger().info('Gripper Controller node started.')
        
        # GripperCommand is a standard action for this purpose.
        self._action_server = ActionServer(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd',
            self.execute_callback)
            
        self.get_logger().info("GripperCommand action server is ready.")

    def execute_callback(self, goal_handle):
        """
        Executes a gripper command goal (open or close).
        """
        command = goal_handle.request.command
        position = command.position
        
        self.get_logger().info(f"Executing gripper command. Target position: {position}")
        
        # In a real robot, you would map the 'position' (typically from 0.0 for
        # closed to 1.0 for open) to the specific joint angles or motor commands
        # for your gripper hardware.
        
        if position > 0.5:
            self.get_logger().info("  Command: OPEN GRIPPER (mock)")
        else:
            self.get_logger().info("  Command: CLOSE GRIPPER (mock)")
            
        # Simulate time to execute
        # time.sleep(1)
        
        self.get_logger().info("Gripper command succeeded (mock).")
        goal_handle.succeed()
        
        result = GripperCommand.Result()
        result.position = position
        result.reached_goal = True
        return result

def main(args=None):
    rclpy.init(args=args)
    gripper_controller = GripperController()
    rclpy.spin(gripper_controller)
    gripper_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
