# Embodied Reasoning: Grounding Language in Physics

This chapter explains how robots use spatial reasoning, affordance understanding, and physical constraints to execute language instructions. The core challenge: language is abstract ("pick up the cup"), but robots live in continuous 3D space with physics constraints.

---

## The Embodied Reasoning Problem

### Why Language Alone Isn't Enough

Consider the instruction: **"Pick up the red cup on the table."**

A human understands this naturally. A language model can parse the syntax:
```
[Action] [Object: color=red, type=cup] [Location: preposition=on, ref=table]
```

But to **execute** this, a robot must answer:

1. **Perception**: Where is the red cup? → Image coordinates (480px, 320px)
2. **Spatial Reasoning**: What's the 3D position? → World frame (0.3m, 0.2m, 0.8m)
3. **Affordance**: How should I grasp it? → Grasp point, approach angle, gripper force
4. **Physics**: Will my motion succeed? → Collision-free path, weight support, friction
5. **Motor Control**: What joint angles? → Inverse kinematics (7 joint values)

**The semantic gap** between "pick up the cup" and [0.45, 0.32, 0.18, -1.2, 0.8, 2.1, 0.05] (joint angles + gripper width) is enormous. This is embodied reasoning.

---

## The Four Pillars of Embodied Reasoning

### Pillar 1: Spatial Grounding (Image → World)

**Problem**: Language refers to scene in image. Robot needs world coordinates.

**Coordinate Frames**:
```
Image Frame:          Robot Base Frame:        Gripper Frame:
(0,0) ────x→          Z↑                       Z↑ (approach direction)
      │               │                        │
      y               │  X→                    Y→ (gripper width)
      ↓               └─Y                      X (depth)

(480px, 320px)        (0.3m, 0.2m, 0.8m)      (0.0, 0.0, 0.0) - gripper center
```

**Transformations**:
```python
def image_to_world(pixel_pos, depth, camera_K, camera_pose):
    """
    Convert image pixel to 3D world coordinate.

    Args:
        pixel_pos: (u, v) in image (480×640)
        depth: Z distance from camera (meters)
        camera_K: 3×3 intrinsic matrix
        camera_pose: 4×4 homogeneous matrix (camera to world)

    Returns:
        point_3d: (x, y, z) in world frame (meters)
    """

    # Step 1: Pixel to camera frame
    u, v = pixel_pos
    fx = camera_K[0, 0]  # focal length x
    fy = camera_K[1, 1]  # focal length y
    cx = camera_K[0, 2]  # principal point x
    cy = camera_K[1, 2]  # principal point y

    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth

    point_cam = np.array([x_cam, y_cam, z_cam, 1.0])

    # Step 2: Camera frame to world frame
    point_world = camera_pose @ point_cam

    return point_world[:3]  # Return (x, y, z)
```

**Key Insight**: A robot seeing "red cup at image pixel (480, 320)" knows its 3D position if it knows the camera calibration and pose. Modern systems use:
- **Depth cameras** (RealSense, Kinect): Direct z measurement
- **SLAM** (Simultaneous Localization & Mapping): Estimate camera pose in real-time
- **Calibration**: Factory calibration for camera intrinsics

### Pillar 2: Affordance Understanding

**Definition**: An **affordance** is a property of an object that suggests an action.

**Examples**:
- Cup handle → "Graspable by handle"
- Flat table surface → "Suitable for placing objects"
- Heavy object → "Requires two-hand grasp or lifting assistance"
- Fragile glass → "Low force, careful handling"

**How Robots Learn Affordances**:

**Approach 1: Explicit Learning**
```python
class AffordancePredictor(nn.Module):
    """Learn affordances from demonstrations."""

    def __init__(self):
        self.vision_encoder = ViT_B32()  # Image features
        self.affordance_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 affordance attributes
        )

    def forward(self, image):
        """
        image: [B, 3, 224, 224]
        returns: affordances dict
        """
        features = self.vision_encoder(image)  # [B, 512]

        # Predict continuous affordances
        affordances_raw = self.affordance_head(features)  # [B, 10]

        return {
            'graspable': sigmoid(affordances_raw[:, 0]),  # 0-1 (is it graspable?)
            'grasp_width': affordances_raw[:, 1:2] * 0.1,  # 0-0.1m
            'grasp_force': affordances_raw[:, 2:3] * 100,  # 0-100N
            'fragile': sigmoid(affordances_raw[:, 3]),     # 0-1 (fragility)
            'weight': affordances_raw[:, 4:5],              # kg
            'contact_point_offset': affordances_raw[:, 5:8],  # 3D offset from center
            'approach_angle': affordances_raw[:, 8:10] * np.pi,  # roll, pitch in radians
        }
```

**Approach 2: Vision-Language Reasoning**
```python
def infer_affordances_with_llm(image, task_description):
    """Use vision-language model to reason about affordances."""

    # Example image: cup on table
    # Example task: "Pick up the cup"

    llm_prompt = f"""
    Image shows: [analyze image]
    Task: {task_description}

    Infer affordances:
    1. Is the object graspable? (yes/no)
    2. Where should I grasp? (description + position)
    3. What force is safe? (force in Newtons)
    4. Is the object fragile? (yes/no)
    5. Approach angle recommendation? (description)

    Respond in JSON format:
    {{
      "graspable": true/false,
      "grasp_position": "handle" or "side" or "top",
      "safe_force_N": 50,
      "fragile": true/false,
      "approach_angle_deg": 45,
      "notes": "..."
    }}
    """

    response = llm(image, llm_prompt)
    affordances = json.loads(response)

    return affordances
```

**Real Example: RT-2 Learning Affordances**

From robot demonstrations, RT-2 learns:
- Different objects require different gripper widths
- Fragile objects require lower forces
- Tall objects need different approach angles
- Novel objects infer affordances from visual similarity

After seeing 100 grasping attempts, the model learns implicit affordance models.

### Pillar 3: Physical Reasoning

**Problem**: Not all actions are physically feasible.

**Constraints**:
```
1. Workspace constraint: Robot reach is limited (e.g., 0.6m)
   Target position must satisfy: ||target - base_position|| ≤ reach

2. Joint limits: Each joint has min/max angles
   θ_1 ∈ [-1.57, 1.57] rad (shoulder pan)
   θ_2 ∈ [0.0, 3.14] rad (shoulder lift)
   ...

3. Collision avoidance: Robot mustn't hit environment
   min_distance(gripper, obstacles) ≥ 0.05m (5cm safety margin)

4. Dynamics constraints: Acceleration limited by motors
   max_acceleration = 2.0 m/s² (max velocity change)
   max_torque = 150 Nm per joint

5. Force constraints: Gripper has force limits
   max_grasp_force = 300N (material yield)
   min_grasp_force = 10N (object not slipping)
```

**Checking Constraints**:
```python
def is_action_feasible(action, robot_state, scene):
    """Check if proposed action is physically feasible."""

    target_pos = action['target_position']
    gripper_force = action['gripper_force']
    approach_duration = action['approach_time']

    # 1. Workspace check
    distance_to_base = np.linalg.norm(target_pos - robot_state.base_position)
    if distance_to_base > 0.6:  # 0.6m reach
        return False, "Target out of workspace"

    # 2. IK check (can we reach this position?)
    try:
        joint_solution = compute_ik(target_pos, preferred_orientation)
    except IKSolver.NoSolutionError:
        return False, "No IK solution"

    # 3. Joint limits check
    for i, theta in enumerate(joint_solution):
        if not (joint_limits[i][0] <= theta <= joint_limits[i][1]):
            return False, f"Joint {i} out of limits"

    # 4. Collision check
    path = plan_trajectory(robot_state, joint_solution)
    collision_free = scene.check_collision_free(path, margin=0.05)
    if not collision_free:
        return False, "Collision with environment"

    # 5. Force check
    if not (10 <= gripper_force <= 300):
        return False, f"Gripper force {gripper_force}N out of range"

    # 6. Dynamics check
    required_acceleration = 2 * distance_to_base / (approach_duration ** 2)
    if required_acceleration > 2.0:  # 2.0 m/s² max
        return False, f"Acceleration {required_acceleration:.1f} m/s² too high"

    return True, "Action is feasible"
```

**Handling Infeasibility**:
```python
def adapt_action_to_constraints(desired_action, robot_state, scene):
    """If action is infeasible, adapt it."""

    target_pos = desired_action['target_position']
    gripper_force = desired_action['gripper_force']
    approach_time = desired_action['approach_time']

    # If out of workspace, clamp to nearest reachable position
    distance = np.linalg.norm(target_pos - robot_state.base_position)
    if distance > 0.6:
        direction = (target_pos - robot_state.base_position) / distance
        target_pos = robot_state.base_position + 0.6 * direction
        print("Clamped to workspace boundary")

    # If force out of range, clip to valid range
    gripper_force = np.clip(gripper_force, 10, 300)

    # If approach time too short, extend it
    if required_acceleration > 2.0:
        approach_time = np.sqrt(2 * distance / 2.0)
        print(f"Extended approach time to {approach_time:.2f}s")

    return {
        'target_position': target_pos,
        'gripper_force': gripper_force,
        'approach_time': approach_time,
        'notes': 'Action adapted to satisfy constraints'
    }
```

### Pillar 4: Planning with Constraints

**Problem**: Multi-step tasks require reasoning about intermediate states.

**Example**: "Stack the cube on the red block, but don't knock over the bottle."

**Decomposition**:
```
Goal: Stack cube on red block (avoid bottle)

└─ Sub-goal 1: Pick up the cube
   ├─ Detect cube position
   ├─ Approach cube (collision-free path)
   ├─ Grasp cube (affordance-based)
   └─ Lift cube (0.3m above table)

└─ Sub-goal 2: Move to red block
   ├─ Detect red block position
   ├─ Plan path that avoids bottle
   │  (Plan around bottle, maintain cube above table level)
   └─ Move to position above red block

└─ Sub-goal 3: Place cube on red block
   ├─ Lower cube gradually
   ├─ Release at correct force
   └─ Withdraw gripper (vertical first, then lateral)

└─ Sub-goal 4: Verify success
    ├─ Cube resting on red block? (stability check)
    ├─ Bottle still intact? (collision detection)
    └─ Grasp released? (force sensor verification)
```

**LLM-Based Planning**:
```python
def plan_task_with_llm(task_description, scene_image, robot_capabilities):
    """Use LLM to decompose task into sub-goals."""

    scene_description = describe_scene(scene_image)
    # "Scene contains: red cube at (0.3, 0.2), red block at (0.5, 0.5),
    #  blue bottle at (0.4, 0.3). Robot gripper at (0, 0, 0.5)."

    prompt = f"""
    Scene: {scene_description}
    Task: {task_description}
    Robot capabilities: {robot_capabilities}

    Decompose this task into step-by-step sub-goals.
    For each sub-goal, specify:
    1. What to do
    2. Why it matters
    3. Success criteria
    4. Potential failure modes

    Format as JSON:
    {{
      "sub_goals": [
        {{
          "step": 1,
          "action": "Pick up the red cube",
          "target_object": "red cube",
          "reason": "Need to move it to the red block",
          "success_criteria": ["Cube lifted 0.3m above table"],
          "failure_modes": ["Cube slips", "Knocks over bottle"]
        }},
        ...
      ],
      "overall_strategy": "...",
      "estimated_success_rate": 0.85
    }}
    """

    plan = llm(prompt)
    return json.loads(plan)
```

---

## Putting It Together: A Complete Embodied Reasoning Loop

```python
class EmbodiedReasoner:
    """
    Integrates spatial grounding, affordance understanding,
    physical reasoning, and planning.
    """

    def __init__(self, robot, scene):
        self.robot = robot
        self.scene = scene
        self.affordance_model = load_affordance_model()
        self.llm = load_language_model()

    def execute_instruction(self, instruction):
        """Execute a natural language instruction."""

        # Step 1: Perceive scene
        rgb_image = self.robot.camera.get_frame()
        depth_image = self.robot.depth_camera.get_frame()
        scene_description = self.perceive_scene(rgb_image, depth_image)

        # Step 2: Plan with LLM
        plan = self.llm(
            image=rgb_image,
            instruction=instruction,
            scene=scene_description
        )
        # Plan: [
        #   {"sub_goal": "Pick up red cube", "target": (0.3, 0.2), ...},
        #   {"sub_goal": "Move to red block", "target": (0.5, 0.5), ...},
        #   {"sub_goal": "Place on red block", "height": 0.25, ...}
        # ]

        # Step 3: Execute sub-goals
        for sub_goal in plan['sub_goals']:
            self.execute_sub_goal(sub_goal, rgb_image, depth_image)

        return {"success": True}

    def perceive_scene(self, rgb, depth):
        """Extract scene description (objects, positions, properties)."""

        # Detect objects with vision model
        detections = self.detect_objects(rgb)
        # [
        #   {"class": "cube", "color": "red", "center_pixel": (480, 320)},
        #   {"class": "block", "color": "red", "center_pixel": (620, 400)},
        #   {"class": "bottle", "color": "blue", "center_pixel": (550, 350)},
        # ]

        # Convert pixel to world coordinates
        for det in detections:
            z = depth[det['center_pixel'][1], det['center_pixel'][0]]
            det['position_3d'] = self.pixel_to_world(
                det['center_pixel'], z
            )

        return detections

    def execute_sub_goal(self, sub_goal, rgb, depth):
        """Execute a single sub-goal."""

        target_object = sub_goal['target']

        # Step 1: Predict affordances
        affordances = self.affordance_model(rgb, target_object)
        # {
        #   "grasp_position": "top",
        #   "grasp_width": 0.08,
        #   "grasp_force": 50,
        #   "approach_angle": [0, 0, 1]  (approach from above)
        # }

        # Step 2: Plan collision-free trajectory
        trajectory = self.plan_trajectory(
            target_object,
            affordances
        )

        # Step 3: Check constraints
        is_feasible, reason = self.check_feasibility(trajectory)
        if not is_feasible:
            print(f"Infeasible: {reason}")
            trajectory = self.adapt_trajectory(trajectory)

        # Step 4: Execute
        self.robot.execute_trajectory(trajectory)

        # Step 5: Verify success
        success = self.verify_success(sub_goal)
        return success
```

---

## Common Embodied Reasoning Patterns

### Pattern 1: Multi-View Reasoning

**Problem**: Single camera may not see all details (occlusion).

**Solution**: Use multiple viewpoints
```python
# Get current view
view1 = robot.camera.get_frame()
detections1 = detect_objects(view1)

# Move slightly and get second view
robot.move_head(pan=0.3)  # Look 30° to the side
view2 = robot.camera.get_frame()
detections2 = detect_objects(view2)

# Fuse detections
fused_objects = fuse_detections(detections1, detections2)
# More confident object positions and properties
```

### Pattern 2: Hypothetical Simulation

**Problem**: Not sure if action will succeed. Need to check before committing.

**Solution**: Simulate before executing
```python
# Simulate the action
simulated_result = sim.execute_action(action)

# Check if simulation succeeded
if simulated_result['success']:
    # Execute on real robot
    real_result = robot.execute_action(action)
else:
    # Try alternative action
    action = adapt_action(action)
    real_result = robot.execute_action(action)
```

### Pattern 3: Iterative Refinement

**Problem**: First attempt may not be perfect. Refine based on feedback.

**Solution**: Perception-action loop
```python
for iteration in range(3):  # Try up to 3 times
    # Get current observation
    obs = robot.get_observation()

    # Reason about what went wrong
    error = llm(f"Image shows: {obs}. We're trying to {task}. What went wrong?")
    # Example error: "Gripper not aligned with handle"

    # Compute corrective action
    correction = llm(f"Error: {error}. How should we adjust?")
    # Example correction: "Rotate gripper 15° clockwise"

    # Apply correction
    robot.execute_correction(correction)

    # Check if now successful
    if verify_success(task, obs):
        break
```

---

## Key Takeaways

| Concept | Definition | Why It Matters |
|---------|-----------|---|
| **Spatial Grounding** | Convert image coordinates to world frame | Robot acts in 3D space, not image space |
| **Affordance** | Property suggesting an action | Guides how to interact with novel objects |
| **Constraint Checking** | Verify action is physically feasible | Prevent failures and collisions |
| **Task Decomposition** | Break complex goal into sub-goals | Manage multi-step reasoning |
| **Multi-View Reasoning** | Combine information from multiple perspectives | Handle occlusions and ambiguity |
| **Iterative Refinement** | Adjust actions based on feedback | Recover from errors |

---

## Next Steps

1. **Learn action grounding** → Read action-grounding.md
2. **See how to implement** → Study vla_architecture.md
3. **Code it** → Run vla_policy_learner.py
4. **Apply to tasks** → Complete exercises with multi-step scenarios

---

**Further Reading**:
- Affordance Learning: https://arxiv.org/abs/1904.01169
- VoxPoser (spatial reasoning): https://arxiv.org/abs/2307.05973
- Self-supervised affordance learning: https://arxiv.org/abs/2303.14910
