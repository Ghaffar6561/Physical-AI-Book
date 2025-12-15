import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from interfaces.msg import ActionPlan # Assuming a custom ActionPlan message

# This would import the actual LLM querying logic
from llm_integration import query_llm_for_plan

class TaskPlanner(Node):
    """
    A node that receives high-level commands, queries an LLM to create a
    plan, and publishes that plan for execution.
    """
    def __init__(self):
        super().__init__('task_planner')
        self.get_logger().info('Task Planner node started.')
        
        # Subscriber to the user command topic
        self.subscription = self.create_subscription(
            String,
            '/user_command',
            self.command_callback,
            10)
            
        # Publisher for the generated action plan
        self.publisher = self.create_publisher(
            ActionPlan,
            '/action_plan',
            10)

    def command_callback(self, msg):
        """
        Callback for processing incoming user commands.
        """
        self.get_logger().info(f"Received command: '{msg.data}'")
        
        # 1. Get the latest world state (e.g., via a service call or topic)
        world_state = self.get_world_state()
        
        # 2. Query the LLM to get a plan
        # This function would contain the prompt engineering logic
        plan_dict = query_llm_for_plan(msg.data, world_state)
        
        if not plan_dict:
            self.get_logger().error("Failed to get a valid plan from the LLM.")
            return
            
        # 3. Convert the plan dictionary into a ROS message
        plan_msg = self.create_plan_message(plan_dict)
        
        # 4. Publish the plan
        self.publisher.publish(plan_msg)
        self.get_logger().info("Published a new action plan.")

    def get_world_state(self):
        """
        Placeholder function to get the current state of the world.
        In a real system, this would subscribe to topics like /robot_pose,
        /detected_objects, etc., and aggregate them.
        """
        return {
            "robot_pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "objects": [
                {"name": "red_ball", "position": [1.0, 2.0, 0.5]}
            ],
            "gripper_empty": True
        }
        
    def create_plan_message(self, plan_dict):
        """
        Converts a list of action dictionaries into a custom ActionPlan message.
        """
        # This assumes you have a custom message `ActionPlan.msg` defined as:
        # interfaces/Action[] steps
        # And `Action.msg` defined as:
        # string name
        # string[] parameters
        
        plan_msg = ActionPlan()
        # Logic to populate plan_msg from plan_dict would go here.
        # For now, this is a conceptual placeholder.
        self.get_logger().info(f"Generated plan message from dict: {plan_dict}")
        return plan_msg

def main(args=None):
    rclpy.init(args=args)
    task_planner = TaskPlanner()
    # You would also need to create and spin the llm_integration node
    rclpy.spin(task_planner)
    task_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
