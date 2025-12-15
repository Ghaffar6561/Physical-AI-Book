# Prompting Strategies for Robotics

## Introduction

Prompting strategies are critical for effectively using Large Language Models (LLMs) in robotics applications. Since most current LLMs are not specifically trained for robotics, careful prompting is essential to generate useful outputs that can be reliably converted into robot actions. This module covers various prompting approaches and best practices for robotics applications.

## Core Prompting Principles for Robotics

### 1. Specificity and Structure

Robotic systems require precise, structured outputs. Effective prompts should specify:
- Expected output format
- Required information elements
- Constraints and limitations

```python
def create_structured_prompt(command, robot_capabilities, environment_state):
    """
    Create a structured prompt for robot task planning.
    
    Args:
        command: Natural language command
        robot_capabilities: List of robot's capabilities
        environment_state: Current state of environment
        
    Returns:
        Structured prompt string
    """
    prompt = f"""
    You are a robot task planner. Convert the human command into a detailed execution plan.

    ROBOT CAPABILITIES: {', '.join(robot_capabilities)}

    CURRENT ENVIRONMENT STATE:
    {environment_state}

    HUMAN COMMAND: {command}

    OUTPUT REQUIREMENTS:
    1. Output must be a numbered sequence of robot actions (maximum 10 steps)
    2. Each action must be executable by the robot capabilities listed above
    3. Include object detection before manipulation
    4. Include navigation actions to reach objects
    5. Format: "1. [action]", "2. [action]", etc.

    ROBOT ACTION PLAN:
    """
    return prompt.strip()

# Example usage
capabilities = ["navigate_to", "detect_object", "grasp_object", "release_object", "open_door"]
env_state = "Kitchen with table, chairs, red cup on table, blue cup in sink"
command = "Bring me the red cup from the table"

prompt = create_structured_prompt(command, capabilities, env_state)
print(prompt)
```

### 2. Context Provisioning

Provide the LLM with relevant context about the robot's situation:

```python
def create_contextual_prompt(robot_type, environment, command):
    """
    Create a prompt with contextual information for specific robot types.
    
    Args:
        robot_type: Type of robot (e.g., "wheeled", "humanoid", "arm_only")
        environment: Description of current environment
        command: User command
        
    Returns:
        Contextual prompt string
    """
    # Different robots have different capabilities and constraints
    context = ""
    if "wheeled" in robot_type.lower():
        context = "The robot is a wheeled mobile robot that navigates using differential drive."
    elif "humanoid" in robot_type.lower():
        context = "The robot is a humanoid with two arms, two legs, and a head with cameras."
    elif "arm" in robot_type.lower():
        context = "The robot is a fixed robotic arm with 7 degrees of freedom."
    
    prompt = f"""
    {context}

    ENVIRONMENT: {environment}

    USER COMMAND: {command}

    PLANNING INSTRUCTIONS:
    - Consider the robot's physical form when planning actions
    - Ensure all actions are achievable by the robot type
    - Account for navigation limitations if mobile
    - Include safety checks appropriate to the robot type

    ACTION SEQUENCE:
    """
    return prompt.strip()
```

## Advanced Prompting Strategies

### 1. Chain-of-Thought Prompting

Guide the LLM to reason step-by-step before providing the final answer:

```python
def create_chain_of_thought_prompt(command, environment):
    """
    Create a prompt that guides the LLM through step-by-step reasoning.
    """
    prompt = f"""
    Plan the following robot task by thinking through it step-by-step:

    USER COMMAND: {command}
    ENVIRONMENT: {environment}

    REASONING PROCESS:
    1. Task Understanding: What is the user asking for?
    2. Environment Analysis: What objects and locations are relevant?
    3. Capability Check: What can the robot do to achieve this?
    4. Step-by-Step Plan: Create a sequence of actions
    5. Safety Check: Are there any safety concerns?

    Let's think step by step:

    1. Task Understanding: The user wants [describe the task]
    2. Environment Analysis: The environment contains [relevant objects/locations]
    3. Capability Check: The robot can [list relevant capabilities]
    4. Step-by-Step Plan:
       - Step 1: [action]
       - Step 2: [action]
       ...
    5. Safety Check: [any safety considerations]

    FINAL ACTION SEQUENCE:
    1. [action]
    2. [action]
    ...
    """
    return prompt.strip()
```

### 2. Few-Shot Learning

Provide examples to guide the LLM's behavior:

```python
def create_fewshot_prompt(command):
    """
    Create a prompt with examples for few-shot learning.
    """
    prompt = """
    You are a robot task planner. Convert natural language commands to robot actions.
    Here are some examples:

    Example 1:
    Command: "Go to the kitchen and bring the apple"
    Robot Actions:
    1. Navigate to kitchen
    2. Detect apple
    3. Grasp apple
    4. Navigate to user
    5. Release apple

    Example 2:
    Command: "Open the door and enter the room"
    Robot Actions:
    1. Navigate to door
    2. Detect door handle
    3. Grasp door handle
    4. Open door
    5. Navigate through doorway

    Example 3:
    Command: "Pick up the red block and place it on the blue mat"
    Robot Actions:
    1. Detect red block
    2. Navigate to red block
    3. Grasp red block
    4. Detect blue mat
    5. Navigate to blue mat
    6. Release red block

    Now, convert this command to robot actions:

    Command: {command}
    Robot Actions:
    """.format(command=command)
    
    return prompt.strip()
```

### 3. Role-Based Prompting

Explicitly assign roles to the LLM:

```python
def create_role_based_prompt(robot_specifications, command):
    """
    Create a prompt that assigns a specific role to the LLM.
    
    Args:
        robot_specifications: Technical specs of the robot
        command: User command
    """
    prompt = f"""
    ROLE: You are an autonomous robot control system with the following specifications:
    {robot_specifications}

    MISSION: Interpret the user command and generate a sequence of executable robot actions.

    USER COMMAND: {command}

    ACTION GENERATION REQUIREMENTS:
    - Only use actions that match the robot's capabilities
    - Ensure action sequence is logical and safe
    - Include necessary sensory feedback steps
    - Format as numbered list

    EXECUTION PLAN:
    """
    return prompt.strip()

# Example robot specifications
robot_specs = """
- Mobility: 2D planar navigation, max speed 1 m/s
- Manipulation: 7-DOF arm with 2-finger gripper, max payload 2kg
- Perception: RGB-D camera, 2D LiDAR, object recognition
- Safety: Emergency stop, collision avoidance
"""
```

## Domain-Specific Prompting Techniques

### 1. Navigation and Path Planning

```python
def create_navigation_prompt(destination, environment_map):
    """
    Create a prompt specialized for navigation planning.
    """
    prompt = f"""
    As a mobile robot navigation planner:

    CURRENT POSITION: [x, y, theta]
    DESTINATION: {destination}
    ENVIRONMENT MAP: {environment_map}

    NAVIGATION PLAN:
    1. Identify obstacles between current position and destination
    2. Plan collision-free path
    3. Consider preferred routes if available
    4. Output sequence of waypoints or movement commands

    PATH TO DESTINATION:
    """
    return prompt.strip()
```

### 2. Object Manipulation

```python
def create_manipulation_prompt(object_description, task):
    """
    Create a prompt specialized for manipulation tasks.
    """
    prompt = f"""
    As a robotic manipulation planner:

    TARGET OBJECT: {object_description}
    MANIPULATION TASK: {task}

    GRASP PLANNING:
    1. Analyze object shape and size
    2. Determine optimal grasp points
    3. Calculate required gripper configuration
    4. Plan approach trajectory
    5. Plan execution sequence

    MANIPULATION SEQUENCE:
    """
    return prompt.strip()
```

### 3. Human-Robot Interaction

```python
def create_interaction_prompt(user_command, user_context):
    """
    Create a prompt for social interaction scenarios.
    """
    prompt = f"""
    As a social robot:

    USER COMMAND: {user_command}
    USER CONTEXT: {user_context}

    INTERACTION PLAN:
    1. Understand user intent
    2. Consider social appropriateness
    3. Plan appropriate response/action
    4. Include politeness markers if needed

    RESPONSE/ACTION:
    """
    return prompt.strip()
```

## Validation and Safety Through Prompting

### 1. Safety-Aware Prompting

```python
def create_safe_action_prompt(command, safety_constraints):
    """
    Create a prompt that emphasizes safety considerations.
    """
    prompt = f"""
    ROBOT SAFETY ASSISTANT

    USER COMMAND: {command}

    SAFETY CONSTRAINTS:
    {safety_constraints}

    Before generating actions, consider:
    1. Collision avoidance with humans and obstacles
    2. Safe operation limits
    3. Emergency procedures
    4. Ethical considerations

    SAFE ACTION SEQUENCE:
    """
    return prompt.strip()
```

### 2. Verification Prompts

Use prompting to validate generated plans:

```python
def create_verification_prompt(action_plan, command, constraints):
    """
    Create a prompt to verify the safety and feasibility of an action plan.
    """
    prompt = f"""
    You are reviewing a robot action plan for safety and feasibility.

    ORIGINAL COMMAND: {command}
    ACTION PLAN: {action_plan}
    CONSTRAINTS: {constraints}

    ANALYSIS REQUIRED:
    1. Does the plan achieve the command goal?
    2. Are all actions physically possible?
    3. Does the plan respect safety constraints?
    4. Are there logical inconsistencies?
    5. What are potential failure points?

    FEEDBACK:
    """
    return prompt.strip()
```

## Error Handling and Fallback Strategies

### 1. Ambiguity Resolution

```python
def create_ambiguity_resolution_prompt(command, scene_description):
    """
    Prompt for resolving ambiguous commands.
    """
    prompt = f"""
    The following command is ambiguous. Identify the ambiguity and suggest clarifications:

    USER COMMAND: {command}
    SCENE DESCRIPTION: {scene_description}

    AMBIGUITY ANALYSIS:
    1. What is ambiguous about this command?
    2. What specific information is needed?
    3. Formulate clarification questions for the user.

    CLARIFICATION QUESTIONS:
    """
    return prompt.strip()
```

### 2. Capability Limitation Handling

```python
def create_fallback_prompt(command, robot_capabilities):
    """
    Generate fallback options when command exceeds robot capabilities.
    """
    prompt = f"""
    The user command exceeds the robot's current capabilities.

    USER COMMAND: {command}
    ROBOT CAPABILITIES: {robot_capabilities}

    ANALYZE THE LIMITATION AND SUGGEST:
    1. Why the command cannot be exactly executed
    2. What is the closest achievable alternative
    3. How to communicate the limitation to the user

    ALTERNATIVE APPROACH:
    """
    return prompt.strip()
```

## Prompt Engineering Best Practices for Robotics

### 1. Iterative Refinement

```python
class PromptRefiner:
    """
    Helper class to iteratively refine prompts based on performance.
    """
    def __init__(self, initial_prompt, evaluation_function):
        self.prompt = initial_prompt
        self.eval_func = evaluation_function
        self.history = []
    
    def refine(self, test_examples, target_performance):
        """
        Refine the prompt based on test performance.
        """
        current_score = self.eval_func(self.prompt, test_examples)
        
        # Simple refinement strategy: add more examples if performance is low
        if current_score < target_performance:
            self.add_examples(test_examples)
            new_score = self.eval_func(self.prompt, test_examples)
            
            if new_score > current_score:
                self.history.append({
                    'prompt': self.prompt,
                    'score': new_score,
                    'improvement': new_score - current_score
                })
                return True
        
        return False
    
    def add_examples(self, examples):
        """
        Add more examples to the prompt.
        """
        # Extract high-performing examples
        good_examples = self.select_good_examples(examples)
        
        # Add to current prompt
        example_text = self.format_examples(good_examples)
        self.prompt = f"{example_text}\n\n{self.prompt}"
    
    def select_good_examples(self, examples):
        """
        Select examples that are most relevant/appropriate.
        """
        # Implementation would analyze examples and select best ones
        return examples[:3]  # Take first 3 as example
    
    def format_examples(self, examples):
        """
        Format examples for inclusion in prompt.
        """
        formatted = "EXAMPLES:\n"
        for i, (input_cmd, output_actions) in enumerate(examples):
            formatted += f"Example {i+1}:\n"
            formatted += f"Input: {input_cmd}\n"
            formatted += f"Output: {output_actions}\n\n"
        
        return formatted
```

### 2. Multi-Perspective Validation

```python
def create_multi_perspective_prompt(command, environment):
    """
    Create a prompt that checks multiple perspectives.
    """
    prompt = f"""
    ANALYZE THE FOLLOWING ROBOT TASK FROM MULTIPLE PERSPECTIVES:

    COMMAND: {command}
    ENVIRONMENT: {environment}

    PERSPECTIVE 1 - TASK ACHIEVEMENT: How to successfully complete the task?
    PERSPECTIVE 2 - SAFETY: How to ensure safe execution?
    PERSPECTIVE 3 - EFFICIENCY: How to minimize time/energy cost?
    PERSPECTIVE 4 - ROBUSTNESS: How to handle potential failures?

    INTEGRATED PLAN:
    1. [Action considering all perspectives]
    2. [Action considering all perspectives]
    ...
    """
    return prompt.strip()
```

## Measuring Prompt Effectiveness

### Evaluation Metrics

```python
def evaluate_prompt_effectiveness(prompt, test_cases):
    """
    Evaluate how effective a prompt is for robotics applications.
    
    Args:
        prompt: The prompt to evaluate
        test_cases: List of (command, expected_output) pairs
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'success_rate': 0,
        'action_validity': 0,
        'semantic_accuracy': 0,
        'format_compliance': 0
    }
    
    correct = 0
    valid_actions = 0
    properly_formatted = 0
    
    for command, expected in test_cases:
        # Generate response using the prompt
        generated = generate_with_prompt(prompt, command)
        
        # Check if action sequence is valid
        if is_valid_action_sequence(generated):
            valid_actions += 1
            
            # Check semantic match with expected
            if semantic_match(generated, expected):
                correct += 1
        
        # Check if output follows expected format
        if follows_format(generated):
            properly_formatted += 1
    
    n_cases = len(test_cases)
    metrics['success_rate'] = correct / n_cases if n_cases > 0 else 0
    metrics['action_validity'] = valid_actions / n_cases if n_cases > 0 else 0
    metrics['format_compliance'] = properly_formatted / n_cases if n_cases > 0 else 0
    
    return metrics

def generate_with_prompt(prompt, command):
    """
    Simulate generation with a given prompt.
    """
    # This would interface with an actual LLM in practice
    return "1. Navigate to target\n2. Execute task"

def is_valid_action_sequence(actions):
    """
    Check if the action sequence is valid.
    """
    # Implement validation logic
    return True

def semantic_match(generated, expected):
    """
    Check if generated actions match expected semantics.
    """
    # Implement semantic matching logic
    return True

def follows_format(actions):
    """
    Check if output follows expected format.
    """
    # Implement format checking logic
    return True
```

## Conclusion

Effective prompting strategies are essential for leveraging LLMs in robotics applications. The key to success lies in:

1. **Structure and Clarity**: Providing clear, structured prompts with specific requirements
2. **Context Awareness**: Including relevant environment and robot information
3. **Safety Focus**: Incorporating safety constraints and validation steps
4. **Iterative Improvement**: Continuously refining prompts based on performance
5. **Domain Specialization**: Adapting prompts to specific robotic capabilities and tasks

By applying these prompting strategies, robotics researchers and engineers can more effectively utilize LLMs to enable natural human-robot interaction and intelligent task planning, while ensuring the outputs are safe, executable, and reliable.