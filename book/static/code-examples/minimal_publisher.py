"""
Minimal ROS 2 Publisher Example

This example demonstrates the simplest possible ROS 2 publisher node.

Key Concepts:
- Creating a ROS 2 node
- Publishing to a topic
- Understanding the publish pattern

Expected Output:
  Publishing data to 'sensor_topic': Hello, ROS 2!
  Publishing data to 'sensor_topic': Hello, ROS 2!
  Publishing data to 'sensor_topic': Hello, ROS 2!

Learning Goals:
- Understand ROS 2 node lifecycle
- See how to publish messages to topics
- Learn about message types
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):
    """A minimal ROS 2 publisher node."""

    def __init__(self):
        super().__init__('minimal_publisher')

        # Create a publisher that publishes String messages to 'sensor_topic'
        # Queue size of 10 means we keep 10 messages in the queue
        self.publisher_ = self.create_publisher(String, 'sensor_topic', 10)

        # Create a timer that calls publish_message() every 0.5 seconds
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.publish_message)

        self.counter = 0

    def publish_message(self):
        """Publish a message to the topic."""
        # Create a String message
        msg = String()
        msg.data = f'Hello, ROS 2! [message #{self.counter}]'

        # Publish the message
        self.publisher_.publish(msg)

        # Log the message (print to console)
        self.get_logger().info(f'Publishing data to "sensor_topic": {msg.data}')

        self.counter += 1


def main():
    """Main function to run the publisher."""
    # Initialize ROS 2
    rclpy.init()

    # Create the node
    minimal_publisher = MinimalPublisher()

    # Keep the node running and processing callbacks
    # In practice, Ctrl+C will stop this
    try:
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        minimal_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
