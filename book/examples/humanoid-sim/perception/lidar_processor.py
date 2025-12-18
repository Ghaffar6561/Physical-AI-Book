import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np

class LidarProcessor(Node):
    """
    A node that processes LiDAR scans to create a simple 2D occupancy grid.
    """
    def __init__(self):
        super().__init__('lidar_processor')
        self.get_logger().info('LiDAR Processor node started.')

        # Subscriber to the LiDAR scan topic
        self.subscription = self.create_subscription(
            LaserScan,
            '/lidar/scan',
            self.scan_callback,
            10)

        # Publisher for the occupancy map
        self.publisher = self.create_publisher(
            OccupancyGrid,
            '/occupancy_map',
            10)
            
        # Occupancy grid parameters (simplified)
        self.map_resolution = 0.1  # meters per cell
        self.map_width = 100       # cells
        self.map_height = 100      # cells
        self.map_origin_x = -5.0   # meters
        self.map_origin_y = -5.0   # meters

    def scan_callback(self, msg):
        """
        Callback for processing incoming LiDAR scans.
        """
        self.get_logger().info('Received a new LiDAR scan.')

        # --- Occupancy Grid Generation Logic Would Go Here ---
        # 1. Initialize an empty map
        grid = np.full((self.map_height, self.map_width), -1, dtype=np.int8) # -1 for unknown

        # 2. For each ray in the scan, calculate the endpoint in the map
        for i, range_val in enumerate(msg.ranges):
            if range_val >= msg.range_max or range_val <= msg.range_min:
                continue
            
            angle = msg.angle_min + i * msg.angle_increment
            x = range_val * np.cos(angle)
            y = range_val * np.sin(angle)
            
            # Convert world coordinates to map coordinates
            map_x = int((x - self.map_origin_x) / self.map_resolution)
            map_y = int((y - self.map_origin_y) / self.map_resolution)
            
            # 3. Mark the cell as occupied
            if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                grid[map_y, map_x] = 100 # 100 for occupied

        # 4. Create and publish the OccupancyGrid message
        grid_msg = self.create_grid_message(grid)
        self.publisher.publish(grid_msg)
        self.get_logger().info('Published new occupancy grid.')

    def create_grid_message(self, grid_data):
        """
        Creates an OccupancyGrid message from the grid data.
        """
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"
        grid_msg.info.resolution = self.map_resolution
        grid_msg.info.width = self.map_width
        grid_msg.info.height = self.map_height
        grid_msg.info.origin.position.x = self.map_origin_x
        grid_msg.info.origin.position.y = self.map_origin_y
        
        # The data is a flattened 1D array
        grid_msg.data = grid_data.flatten().tolist()
        return grid_msg

def main(args=None):
    rclpy.init(args=args)
    lidar_processor = LidarProcessor()
    rclpy.spin(lidar_processor)
    lidar_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
