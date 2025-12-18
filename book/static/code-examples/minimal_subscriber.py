"""
Minimal ROS 2 Subscriber Example

This example demonstrates the simplest possible ROS 2 subscriber node.

Key Concepts:
- Creating a ROS 2 node
- Subscribing to a topic
- Receiving and processing messages

Expected Output:
  I heard: "Hello, ROS 2! [message #0]"
  I heard: "Hello, ROS 2! [message #1]"
  I heard: "Hello, ROS 2! [message #2]"

Learning Goals:
- Understand message subscription pattern
- See how callbacks work in ROS 2
- Learn the flow of data between nodes
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):
    """A minimal ROS 2 subscriber node."""

    def __init__(self):
        super().__init__('minimal_subscriber')

        # Create a subscription to 'sensor_topic'
        # This node will receive messages published to that topic
        # When a message arrives, listener_callback() is called
        self.subscription = self.create_subscription(
            String,
            'sensor_topic',
            self.listener_callback,
            10  # Queue size
        )

        # Prevent unused variable warning
        self.subscription

    def listener_callback(self, msg):
        """Callback function that runs when a message is received."""
        # This function is called automatically whenever a new message arrives
        self.get_logger().info(f'I heard: "{msg.data}"')


def main():
    """Main function to run the subscriber."""
    # Initialize ROS 2
    rclpy.init()

    # Create the node
    minimal_subscriber = MinimalSubscriber()

    # Keep the node running and waiting for messages
    # In practice, Ctrl+C will stop this
    try:
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        minimal_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
