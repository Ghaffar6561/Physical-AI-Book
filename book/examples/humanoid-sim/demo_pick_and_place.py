import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class DemoPickAndPlace(Node):
    """
    A simple node to publish a command to initiate a pick-and-place task.
    This simulates a user or a high-level mission planner triggering the VLA pipeline.
    """
    def __init__(self):
        super().__init__('demo_pick_and_place')
        self.publisher = self.create_publisher(String, '/user_command', 10)
        self.get_logger().info('Demo Pick and Place node started. Will publish command in 5 seconds.')

        # Wait for the rest of the system to be ready before publishing
        self.create_timer(5.0, self.publish_command)

    def publish_command(self):
        """
        Publishes the pick-and-place command.
        """
        msg = String()
        msg.data = "Pick up the red ball and place it on the blue square."
        self.publisher.publish(msg)
        self.get_logger().info(f"Published command: '{msg.data}'")
        
        # We can destroy the timer so it only runs once
        self.destroy_timer(self.get_timers()[0])
        
        # The node can exit after publishing
        self.get_logger().info("Command published. Shutting down demo node.")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    demo_node = DemoPickAndPlace()
    # We spin the node so it can publish its message
    rclpy.spin(demo_node)
    demo_node.destroy_node()

if __name__ == '__main__':
    main()
