# Module 4 Exercises: Vision-Language-Action Systems

These exercises are designed to test your understanding of how to design, build, and debug Vision-Language-Action (VLA) systems.

## Exercise 1: Design a VLA Pipeline

**Scenario**: You have a household robot that needs to perform a multi-step task: "After you've watered the plant in the living room, can you turn on the kitchen lights?"

**Task**:

1.  Draw a diagram of the full VLA pipeline required to handle this command.
2.  For each stage of the pipeline (Speech Recognition, LLM Planning, Execution), describe the specific inputs and outputs.
3.  What challenges does the phrase "After you've watered the plant" introduce for the LLM planner? How might you design your prompt to handle this kind of sequential, conditional command?

## Exercise 2: Engineer Prompts for Task Decomposition

**Scenario**: Your robot has the following action dictionary:
```json
{
    "move_to(location)": "Navigate to a location ('living_room', 'kitchen').",
    "find_object(object_name)": "Locate a specific object.",
    "pick_up(object_name)": "Grasp the specified object.",
    "toggle_switch(switch_name, state)": "Turn a switch 'on' or 'off'."
}
```

**Task**:

1.  Write a complete few-shot prompt to an LLM to decompose the command: "Find the TV remote in the living room and turn it off."
2.  Write a prompt for a more complex command: "Can you see if the stove is on in the kitchen and turn it off if it is?" (Note: The robot doesn't have a direct `is_on` action. The LLM will need to infer a sequence of actions to check this).
3.  How would you modify the prompt to prevent the robot from performing a dangerous action, like `pick_up('stove')`?

## Exercise 3: Trace a Spoken Command

**Scenario**: A user says, **"Hey robot, please grab my keys from the entryway table."**

**Task**:

Trace this command through each stage of the VLA pipeline you've learned about. For each step, describe the data and what is happening to it.

1.  **Speech Recognition**:
    *   Input: Raw audio waveform.
    *   Output: ?
2.  **LLM Planner**:
    *   Input: The text from the previous stage.
    *   Output (as a structured plan): ?
3.  **Action Executor**:
    *   Input: The structured plan from the LLM.
    *   Describe the sequence of (mock) action calls it would make.
4.  **Feedback**:
    *   What happens if the `grasp('keys')` action fails because the robot's camera can't find the keys? How should the Action Executor respond, and what information should it send back to the LLM Planner for re-planning?

---
*Solutions and further discussion can be found in `exercises-answers.md`.*
