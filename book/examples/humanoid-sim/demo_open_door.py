import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class DemoOpenDoor(Node):
    """
    A simple node to publish a command to initiate an open-door task.
    """
    def __init__(self):
        super().__init__('demo_open_door')
        self.publisher = self.create_publisher(String, '/user_command', 10)
        self.get_logger().info('Demo Open Door node started. Will publish command in 5 seconds.')

        self.create_timer(5.0, self.publish_command)

    def publish_command(self):
        """
        Publishes the open-door command.
        """
        msg = String()
        # This is a more complex command that implies finding a handle,
        # manipulating it, and applying force. A real system would need
        # more sophisticated actions to handle this.
        msg.data = "Go to the door and open it."
        self.publisher.publish(msg)
        self.get_logger().info(f"Published command: '{msg.data}'")
        
        self.destroy_timer(self.get_timers()[0])
        
        self.get_logger().info("Command published. Shutting down demo node.")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    demo_node = DemoOpenDoor()
    rclpy.spin(demo_node)
    demo_node.destroy_node()

if __name__ == '__main__':
    main()
