# Prompting Strategies for VLA Systems

This document visualizes different approaches to prompt engineering for language models in robotic tasks.

---

## Strategy 1: Zero-Shot Prompting

**Simplest**: Ask without examples

```
┌─────────────────────────────────────────┐
│ Prompt:                                 │
├─────────────────────────────────────────┤
│ Task: "Pick up the red cup on the left" │
│                                         │
│ Image: [RGB camera frame]               │
│                                         │
│ Analyze the scene and generate          │
│ a detailed action plan.                 │
└─────────────────────────────────────────┘

Result:
Success rate: 30-40% (low, LLM guesses)

Best for:
- Simple, common tasks
- When you have no examples
- Quick prototyping
```

---

## Strategy 2: Few-Shot Prompting

**Better**: Show a few examples

```
┌──────────────────────────────────────────┐
│ Prompt:                                  │
├──────────────────────────────────────────┤
│ Example 1:                               │
│ Task: "Pick up the blue cube"           │
│ Image: [similar scene]                   │
│ Action:                                  │
│   Target position: (0.4, 0.3, 0.9)      │
│   Gripper width: 0.07m                   │
│   Approach height: 0.15m                 │
│   Force: 45N                             │
│                                          │
│ Example 2:                               │
│ Task: "Pick up the green sphere"        │
│ Image: [different scene]                 │
│ Action:                                  │
│   Target position: (0.5, 0.1, 0.85)     │
│   Gripper width: 0.09m                   │
│   Approach height: 0.12m                 │
│   Force: 40N                             │
│                                          │
│ Now perform: "Pick up the red cylinder" │
└──────────────────────────────────────────┘

Result:
Success rate: 60-75% (better, LLM learns from examples)

Best for:
- Novel objects similar to examples
- When you have 2-5 demonstrations
- Balancing simplicity and performance
```

---

## Strategy 3: Chain-of-Thought Prompting

**Structured reasoning**: Make the model think step-by-step

```
┌────────────────────────────────────────────────┐
│ Prompt:                                        │
├────────────────────────────────────────────────┤
│ Task: "Stack the cube on the red block"       │
│ Image: [scene with cube, red block, bottle]   │
│                                                │
│ Let's think step by step:                      │
│                                                │
│ 1. What objects are in the scene?              │
│    - Green cube at (0.3, 0.2, 0.85)           │
│    - Red block at (0.5, 0.5, 0.9)             │
│    - Blue bottle at (0.4, 0.3, 0.8)           │
│                                                │
│ 2. What's the goal?                            │
│    - Move cube to the top of red block        │
│    - Final position: (0.5, 0.5, 1.1)          │
│                                                │
│ 3. What constraints exist?                     │
│    - Avoid knocking over the bottle           │
│    - Gripper force ≤ 50N (not crush)          │
│    - Path must be collision-free              │
│                                                │
│ 4. What's my action plan?                      │
│    Step 1: Grasp cube from top                │
│    Step 2: Lift 0.3m above table              │
│    Step 3: Move to above red block (avoid)    │
│    Step 4: Lower gently and place             │
│                                                │
│ 5. What could go wrong?                        │
│    - Cube too heavy for gripper               │
│    - Trajectory collision with bottle         │
│    - Red block unstable (cube tips)           │
│                                                │
│ Therefore, the action is: [...]               │
└────────────────────────────────────────────────┘

Result:
Success rate: 75-85% (much better, reasoning improves accuracy)

Best for:
- Complex, multi-step tasks
- When you need interpretability
- Handling constraints and ambiguity
```

---

## Strategy 4: Task-Specific Prompting

**Customized**: Tailor the prompt to your robot's specifics

```
┌─────────────────────────────────────────────────┐
│ System Prompt:                                  │
├─────────────────────────────────────────────────┤
│ You are a robot control system.                │
│ Your robot (7-DOF manipulator) is controlled   │
│ by sending 3D target positions and parameters. │
│                                                │
│ Robot specifications:                          │
│ - Reach: 0.6m                                  │
│ - Max joint velocity: 1.5 rad/s                │
│ - Gripper force range: 10-300N                 │
│ - Operating workspace: [−0.6, +0.6] × X       │
│                        [−0.6, +0.6] × Y       │
│                        [0.5, 1.8] × Z          │
│                                                │
│ When responding:                               │
│ 1. Verify target is within workspace           │
│ 2. Predict gripper width based on object       │
│ 3. Choose approach angle from top/side/front   │
│ 4. Estimate force needed (5-100N)              │
│ 5. Respond in JSON format (see examples)       │
│                                                │
│ Output format:                                 │
│ {                                              │
│   "target_position": [x, y, z],               │
│   "gripper_width_m": 0.08,                    │
│   "grasp_force_N": 50,                        │
│   "approach_angle_deg": 45,                   │
│   "approach_height_m": 0.15,                  │
│   "speed_ms": 0.5,                            │
│   "confidence": 0.85,                         │
│   "failure_modes": ["..."]                    │
│ }                                              │
│                                                │
│ Example:                                       │
│ Task: "Pick up the cup"                       │
│ Image: [cup on table]                         │
│ Response:                                      │
│ {                                              │
│   "target_position": [0.3, 0.2, 0.8],        │
│   "gripper_width_m": 0.07,                   │
│   "grasp_force_N": 40,                       │
│   "approach_angle_deg": 0,                   │
│   "approach_height_m": 0.10,                 │
│   "speed_ms": 0.3,                           │
│   "confidence": 0.92,                        │
│   "failure_modes": ["cup too near edge",    │
│                     "handle fragile"]        │
│ }                                              │
└─────────────────────────────────────────────────┘

Result:
Success rate: 80-90% (best for known robot, constrains outputs)

Best for:
- Production deployment on specific robot
- When you need error checking
- Outputting structured, verifiable data
```

---

## Strategy 5: Hierarchical Prompting

**Two-stage**: First planning, then control

```
STAGE 1: Task Planning (Complex, can be slow)
┌──────────────────────────────────────────┐
│ Prompt to Planner (GPT-4):              │
├──────────────────────────────────────────┤
│ Task: "Serve a drink by pouring water   │
│        from pitcher to cup on table"     │
│ Image: [scene with pitcher, cup, table] │
│                                          │
│ Decompose into sub-goals. Be specific   │
│ about locations.                         │
│                                          │
│ Response:                                │
│ Sub-goals:                               │
│ 1. Move to pitcher (location: 0.4, 0.2) │
│ 2. Grasp pitcher handle                 │
│ 3. Lift pitcher to height 1.5m          │
│ 4. Move to above cup (location: 0.5, 0) │
│ 5. Tilt pitcher 45° (pour motion)       │
│ 6. Return pitcher to table              │
└──────────────────────────────────────────┘
           ↓ Sub-goals feed into:

STAGE 2: Sub-Goal Execution (Fast reactive control)
┌──────────────────────────────────────────┐
│ Prompt to Controller (Llama-7B):        │
├──────────────────────────────────────────┤
│ Current state: At rest                   │
│ Sub-goal: "Move to pitcher handle"      │
│ Target: (0.4, 0.2, 0.9)                 │
│                                          │
│ Current position: (0, 0, 0.5)           │
│ Distance: 0.5m                          │
│ Time budget: 5 seconds                  │
│                                          │
│ What joint velocities [rad/s] to reach? │
│                                          │
│ Response:                                │
│ [0.3, 0.2, 0.15, 0, 0.1, -0.2, 0.05]  │
│ (Faster approach toward target)         │
└──────────────────────────────────────────┘

Result:
Success rate: 85-95% (interpretable + fast execution)

Best for:
- Complex, multi-step manipulation
- Latency-tolerant planning (1-2s OK)
- Real-time control required (50-100ms)
```

---

## Comparison: Prompting Strategies

```
Strategy              Accuracy  Speed   Complexity  Data Needed
─────────────────────────────────────────────────────────────
Zero-shot            30-40%    Fast    Low         None
Few-shot             60-75%    Fast    Low         5-10 examples
Chain-of-thought     75-85%    Medium  Medium      5-10 examples
Task-specific        80-90%    Fast    High        Robot specs
Hierarchical         85-95%    Slow*   Medium      100 examples
──────────────────────────────────────────────────────────────
*Slow planning is acceptable; control is fast
```

---

## Best Practices

### 1. **Clarity is Key**
```
Bad:   "Pick up stuff"
Good:  "Pick up the red plastic cup at position (0.3, 0.2, 0.8)"
```

### 2. **Provide Context**
```
Bad:   "Move to target"
Good:  "Move the gripper to (0.5, 0.5, 1.0) at 0.5 m/s,
        avoiding the blue bottle at (0.4, 0.3)"
```

### 3. **Set Constraints Explicitly**
```
Bad:   "Grasp gently"
Good:  "Grasp with 30-50N force (fragile object).
        Max gripper width 0.08m"
```

### 4. **Use Structured Output**
```
Bad:   "Tell me what to do"
Good:  "Respond in JSON with: target_position, gripper_width, force"
```

### 5. **Include Verification**
```
Bad:   "Execute action"
Good:  "Execute action. Then verify success by checking if
        the object is at the target location"
```

---

## Real Example: Full Prompting Pipeline

```python
def generate_robot_action(image, task, robot_specs):
    """Complete prompting pipeline."""

    # Step 1: Perception
    scene_description = describe_scene(image, robot_specs)
    # Output: "Scene contains: red cup at (0.3, 0.2, 0.8),
    #         blue bottle at (0.4, 0.3, 0.8)"

    # Step 2: Task-specific planning prompt
    planning_prompt = f"""
    Robot Specifications:
    {format_robot_specs(robot_specs)}

    Scene Analysis:
    {scene_description}

    Task: {task}

    Decompose into sub-goals using the provided format.
    Ensure all targets are within workspace.
    """

    plan = gpt4(planning_prompt)  # Slow, but thorough (2-3 seconds)

    # Step 3: Execute each sub-goal with reactive controller
    actions = []
    for subgoal in plan['sub_goals']:
        action_prompt = f"""
        You are a fast reactive controller.

        Sub-goal: {subgoal['action']}
        Target: {subgoal['target_position']}

        Current state: [proprioception data]

        Output joint velocities [rad/s].
        Respond in JSON: {{"joint_velocities": [...]}}
        """

        action = llama7b(action_prompt)  # Fast (50ms)
        actions.append(action)

    return actions
```

---

## Key Takeaways

| Strategy | When to Use | Success Rate |
|----------|---|---|
| **Zero-shot** | Prototyping only | 30-40% |
| **Few-shot** | Quick adaptation | 60-75% |
| **Chain-of-thought** | Reasoning needed | 75-85% |
| **Task-specific** | Production, known robot | 80-90% |
| **Hierarchical** | Complex tasks | 85-95% |

---

**Next**: Read vla-architecture.md for system design patterns
