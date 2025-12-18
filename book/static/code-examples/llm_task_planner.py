"""
This file provides a code example for using a Large Language Model (LLM)
to decompose a natural language command into a sequence of robot actions.

Learning Goals:
- Understand how to structure a prompt for robotic task planning.
- See how to parse the LLM's output into a structured format.
- Learn about the importance of an action dictionary and few-shot examples.
"""

import json

# In a real application, you would use a library like `openai` or `huggingface_hub`
# to interact with an LLM. Here, we will simulate the LLM's response.
def query_llm(prompt):
    """
    Simulates a query to an LLM. In a real implementation, this function
    would make an API call.
    """
    print("--- Sending Prompt to LLM ---")
    print(prompt)
    print("-----------------------------")
    
    # This is a mocked response. A real LLM might be less predictable.
    mocked_response = """
[ 
    {"action": "navigate", "parameters": {"location": "kitchen_table"}},
    {"action": "grasp", "parameters": {"object_name": "apple"}},
    {"action": "navigate", "parameters": {"location": "user"}},
    {"action": "place", "parameters": {"location": "user"}}
]    """
    print(f"--- Mocked LLM Response ---\n{mocked_response}\n---------------------------")
    return mocked_response

def get_planning_prompt(command, action_dictionary):
    """
    Constructs a prompt to send to the LLM for task decomposition.
    """
    
    prompt = f"""
You are an AI assistant for a robot operating in a home environment. Your task is to decompose a user's command into a sequence of executable actions.

You must respond with a JSON-formatted list of actions.

Here is the dictionary of available actions:
{json.dumps(action_dictionary, indent=4)}

---\nHere are a few examples of decompositions:

User command: \"Get me the water bottle from the counter.\"
[
    {{"action": "navigate", "parameters": {{"location": "kitchen_counter"}}}},
    {{"action": "grasp", "parameters": {{"object_name": "water_bottle"}}}},
    {{"action": "navigate", "parameters": {{"location": "user"}}}}
]

User command: \"Say hello.\"
[
    {{"action": "say", "parameters": {{"message": "Hello!"}}}}
]
---

Now, decompose the following user command.

User command: "{command}"
    "".strip()
    return prompt

def parse_llm_response(response_text):
    """
    Parses the JSON string from the LLM into a Python list of actions.
    Includes basic validation.
    """
    try:
        plan = json.loads(response_text)
        if not isinstance(plan, list):
            print("Error: LLM response is not a list.")
            return None
        return plan
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from LLM response.")
        return None

def main():
    """
    Main function to demonstrate the LLM task planning process.
    """
    # This action dictionary defines the robot's capabilities.
    # It's crucial for grounding the LLM's planning.
    action_dictionary = {
        "navigate": {
            "description": "Move to a specific location.",
            "parameters": {"location": "string"}
        },
        "grasp": {
            "description": "Pick up an object.",
            "parameters": {"object_name": "string"}
        },
        "place": {
            "description": "Place the held object at a location.",
            "parameters": {"location": "string"}
        },
        "say": {
            "description": "Speak a message.",
            "parameters": {"message": "string"}
        }
    }
    
    user_command = "Please get the apple from the kitchen table and bring it to me."
    
    # 1. Construct the prompt
    prompt = get_planning_prompt(user_command, action_dictionary)
    
    # 2. Query the LLM
    llm_response = query_llm(prompt)
    
    # 3. Parse the response
    action_plan = parse_llm_response(llm_response)
    
    if action_plan:
        print("\n--- Successfully Parsed Action Plan ---")
        for i, step in enumerate(action_plan):
            print(f"Step {i+1}: {step['action']}({step.get('parameters', {})})")
        print("------------------------------------")
        # In a ROS 2 system, this action_plan would be published to the
        # action_executor node.

if __name__ == '__main__':
    main()
