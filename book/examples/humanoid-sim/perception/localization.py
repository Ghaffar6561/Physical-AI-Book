import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf_transformations

class LocalizationNode(Node):
    """
    A simple localization node that uses IMU data to estimate the robot's pose.
    In a real system, this would be a much more complex filter (like a Kalman filter)
    that would also incorporate wheel odometry, visual odometry, etc.
    """
    def __init__(self):
        super().__init__('localization_node')
        self.get_logger().info('Localization Node started.')

        # Subscriber to the IMU data topic
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)
            
        # Publisher for the estimated robot pose
        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            '/robot_pose',
            10)
            
        # Internal state
        self.current_pose = PoseWithCovarianceStamped()
        self.current_pose.header.frame_id = "odom"
        self.get_logger().info('Initialized pose to origin.')

    def imu_callback(self, msg):
        """
        Callback for processing incoming IMU data.
        This is a highly simplified integration for demonstration.
        """
        self.get_logger().info('Received new IMU data.')

        # --- Simple Pose Estimation Logic ---
        # This is a placeholder for a real localization algorithm like EKF or AMCL.
        # We will just use the orientation from the IMU directly.
        
        self.current_pose.header.stamp = self.get_clock().now().to_msg()
        
        # We directly use the orientation from the IMU
        self.current_pose.pose.pose.orientation = msg.orientation
        
        # In a real system, you would integrate linear acceleration to get
        # position, but this is noisy and prone to drift without other sensors.
        # For this example, we'll keep the position at the origin.
        
        # Publish the updated pose
        self.publisher.publish(self.current_pose)
        self.get_logger().info('Published updated robot pose.')

def main(args=None):
    rclpy.init(args=args)
    localization_node = LocalizationNode()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
