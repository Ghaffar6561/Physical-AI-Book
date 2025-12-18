"This file provides a code example for an "Action Executor" node. Its role
is to take a structured action plan (from the LLM planner) and execute it
by calling the appropriate ROS 2 action servers.

Learning Goals:
- Understand the role of an executor in a VLA pipeline.
- See the conceptual structure of a ROS 2 action client.
- Learn how to execute a sequence of actions and handle basic success/failure.
"

import time

# This is a mock of a ROS 2 action client. In a real ROS 2 application,
you would use the `rclpy.action.ActionClient` class.
class MockActionClient:
    def __init__(self, action_name):
        self._action_name = action_name
        print(f"[*] MockActionClient for '{action_name}' initialized.")

    def send_goal(self, goal):
        print(f"[*] Sending goal to '{self._action_name}' action server: {goal}")
        # Simulate the time it takes for the action to complete.
        time.sleep(2)
        
        # Simulate a successful result. In a real scenario, this could fail.
        print(f"[*] Goal for '{self._action_name}' succeeded.")
        return True

class ActionExecutor:
    """
    Executes a plan by calling the corresponding action servers.
    """
    def __init__(self):
        # In a real ROS 2 node, you would create action clients here.
        self.action_clients = {
            "navigate": MockActionClient("navigate"),
            "grasp": MockActionClient("grasp"),
            "place": MockActionClient("place"),
            "say": MockActionClient("say"),
        }
        print("\nActionExecutor initialized with mock action clients.")

    def execute_plan(self, plan):
        """
        Iterates through a plan and executes each action sequentially.

        Args:
            plan: A list of dictionaries, where each dictionary represents an action.
        
        Returns:
            True if the entire plan was executed successfully, False otherwise.
        """
        print("\n--- Starting Execution of Action Plan ---")
        for i, step in enumerate(plan):
            action_name = step.get("action")
            parameters = step.get("parameters", {})
            
            print(f"\nStep {i+1}: Executing {action_name}({parameters})")
            
            client = self.action_clients.get(action_name)
            if not client:
                print(f"  [Error] Unknown action '{action_name}'. Skipping.")
                continue

            # This is a simplified call. Real action clients have more complex
            # goal handling with feedback and result callbacks.
            success = client.send_goal(parameters)
            
            if not success:
                print(f"  [Error] Action '{action_name}' failed. Aborting plan.")
                return False
        
        print("\n--- Action Plan Execution Complete ---")
        return True

def main():
    """
    Main function to demonstrate the action executor.
    """
    # This is a sample plan that would be received from the llm_task_planner
    sample_plan = [
        {"action": "navigate", "parameters": {"location": "kitchen_table"}},
        {"action": "grasp", "parameters": {"object_name": "apple"}},
        {"action": "navigate", "parameters": {"location": "user"}},
        {"action": "place", "parameters": {"location": "user"}}
    ]
    
    executor = ActionExecutor()
    executor.execute_plan(sample_plan)

if __name__ == '__main__':
    main()
