import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from interfaces.msg import ActionPlan, Action
from interfaces.action import Navigate, Grasp, Place # Assuming custom actions

class VLA_ActionExecutor(Node):
    """
    This node subscribes to a validated action plan and executes it by calling
    the appropriate ROS 2 action servers. It manages the sequence of actions,
    waiting for each one to complete before starting the next.
    """
    def __init__(self):
        super().__init__('vla_action_executor')
        self.get_logger().info('VLA Action Executor node started.')
        
        self.subscription = self.create_subscription(
            ActionPlan,
            '/validated_action_plan',
            self.plan_callback,
            10)
            
        # Action clients to interact with the robot's control modules
        self._navigate_client = ActionClient(self, Navigate, '/navigate_to_goal')
        self._grasp_client = ActionClient(self, Grasp, '/grasp_object')
        self._place_client = ActionClient(self, Place, '/place_object')
        
        self.action_clients = {
            "navigate": self._navigate_client,
            "grasp": self._grasp_client,
            "place": self._place_client
        }
        self.get_logger().info("Action clients are ready and waiting for servers.")
        # In a real app, you would wait for servers to be ready before processing plans.

    def plan_callback(self, msg):
        """
        Callback to execute an entire action plan.
        """
        self.get_logger().info(f"Received a new plan with {len(msg.steps)} steps. Starting execution.")
        
        for i, action_step in enumerate(msg.steps):
            self.get_logger().info(f"Executing step {i+1}: {action_step.name}")
            
            client = self.action_clients.get(action_step.name)
            if not client:
                self.get_logger().error(f"Unknown action '{action_step.name}'. Aborting plan.")
                return
            
            # --- This is a simplified, blocking execution loop ---
            # A more robust implementation would use async/await with futures
            
            goal_msg = self.create_goal_msg(action_step)
            if not goal_msg:
                self.get_logger().error("Could not create goal message. Aborting.")
                return

            self.get_logger().info("Sending goal...")
            future = client.send_goal_async(goal_msg)
            
            # This is a blocking wait for the goal to be accepted
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()

            if not goal_handle.accepted:
                self.get_logger().error('Goal was rejected. Aborting plan.')
                return

            self.get_logger().info('Goal accepted. Waiting for result...')
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            
            result = result_future.result().result
            if not result.success:
                self.get_logger().error("Action failed. Aborting plan.")
                return
                
            self.get_logger().info("Action succeeded.")

        self.get_logger().info("Successfully executed all steps in the plan.")

    def create_goal_msg(self, action_step):
        """
        Creates a ROS 2 action goal message from an Action message.
        (Conceptual)
        """
        if action_step.name == "navigate":
            goal_msg = Navigate.Goal()
            # goal_msg.location = action_step.parameters['location']
            return goal_msg
        # ... and so on for other actions
        return None

def main(args=None):
    rclpy.init(args=args)
    action_executor = VLA_ActionExecutor()
    rclpy.spin(action_executor)
    action_executor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
