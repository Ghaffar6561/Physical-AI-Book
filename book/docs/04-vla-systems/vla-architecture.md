# VLA System Architecture & Design Patterns

This chapter shows how to assemble a complete Vision-Language-Action system. You'll learn proven architectural patterns from Google RT-2, CMU VoxPoser, UC Berkeley Code-as-Policies, and others.

---

## The Three Core Architectures

### Architecture 1: Hierarchical (Fast Planning + Reactive Control)

```
┌─────────────────────────────────────────────────┐
│ SEMANTIC PLANNING (LLM, slow OK ~1-2 seconds)   │
│                                                 │
│  Image: [RGB frame]                             │
│  Task: "Pick up the red cup"                    │
│         ↓                                       │
│  LLM reasoning:                                 │
│  "I see a red cup at (0.3m, 0.2m, 0.8m)        │
│   I will grasp from top at 50N force"          │
│         ↓                                       │
│  Output: Sub-goals                              │
│  - Approach: (0.3, 0.2, 0.95) at 0.5 m/s      │
│  - Grasp: width=0.08m, force=50N               │
│  - Lift: 0.3m above table                      │
│  - Place: At target location                    │
└──────────────────┬───────────────────────────────┘
                   │ Sub-goals + current state
                   ↓
┌─────────────────────────────────────────────────┐
│ REACTIVE CONTROL (Neural Network, fast ~50ms)   │
│                                                 │
│  Input: Current observation (proprioception)    │
│  Sub-goal: "Approach target (0.3, 0.2, 0.95)"  │
│         ↓                                       │
│  Policy:                                        │
│  "Current gripper at (0.2, 0.1, 0.5)           │
│   Distance to target: 0.5m                      │
│   Direction: +X, +Y, +Z                        │
│   Publish: [0.3, 0.2, 0.2, 0.1, 0, 0.15]      │
│   (joint velocity commands)"                    │
│         ↓                                       │
│  Execute on robot                              │
└─────────────────────────────────────────────────┘

Advantages:
✓ LLM reasoning is interpretable
✓ Reactive control is fast and robust
✓ Can handle dynamic changes (object moved)
✓ Modular: reuse planning or control

Disadvantages:
✗ Two networks to maintain
✗ Semantic loss at planning-control interface
✗ Sub-goal specification must be precise
```

### Architecture 2: End-to-End (Single Network)

```
┌─────────────────────────────────────────────────┐
│ UNIFIED VLA POLICY                              │
│ (Large neural network, 7B-70B parameters)      │
│                                                 │
│  Input:                                         │
│  - RGB image (480×640×3)                       │
│  - Instruction: "Pick up red cup"              │
│  - Proprioception: [joint angles, gripper]     │
│         ↓                                       │
│  Vision transformer: Extract image features    │
│         ↓                                       │
│  Language model: Encode instruction            │
│         ↓                                       │
│  Fusion network: Combine modalities            │
│         ↓                                       │
│  Action decoder: Predict motor commands        │
│         ↓                                       │
│  Output: [θ₁, θ₂, ..., θ₇, gripper_width]    │
└─────────────────────────────────────────────────┘

Advantages:
✓ Single inference: simpler to deploy
✓ No bottleneck at intermediate representations
✓ Can learn implicit affordances and physics
✓ Fewer hyperparameters to tune

Disadvantages:
✗ Requires massive training data (100K-1M demos)
✗ Black box: hard to debug
✗ Poor generalization to new objects/tasks
✗ Expensive training (weeks of GPU time)
```

### Architecture 3: Modular Reasoning (Code-as-Policies Style)

```
┌──────────────────────────────────┐
│ LANGUAGE MODEL CODE GENERATOR    │
│                                  │
│  Task: "Stack cube on red block" │
│         ↓                        │
│  Generate Python code:           │
│  ┌────────────────────────────┐  │
│  │ cube = find_object("cube") │  │
│  │ red_block = find_object(   │  │
│  │   "red", "block")          │  │
│  │ move_to_grasp(cube)        │  │
│  │ grasp(force=50)            │  │
│  │ move_to(red_block, above)  │  │
│  │ place()                    │  │
│  └────────────────────────────┘  │
│         ↓                        │
│  Execute generated program      │
│  - find_object(): Vision model   │
│  - move_to_grasp(): Learned IK   │
│  - grasp(): Skill from library   │
│  - place(): Learned skill        │
└──────────────────────────────────┘

Advantages:
✓ Interpretable: see the generated code
✓ Composable: reuse functions across tasks
✓ Extensible: add new functions
✓ Verifiable: can check if code is safe

Disadvantages:
✗ Limited to pre-defined APIs
✗ Requires good function library design
✗ Hallucinations in generated code
✗ Error handling complex
```

---

## Choosing Your Architecture

```
Start here: What are your constraints?
    │
    ├─ If: "I have 100K+ robot hours of data"
    │  → Use End-to-End
    │     (Single model, best accuracy, massive training)
    │
    ├─ If: "I have 100-1000 demos, need interpretability"
    │  → Use Hierarchical
    │     (Fast planning + reactive control, modular)
    │
    └─ If: "I need to compose behaviors, have good primitives"
       → Use Modular
          (Code generation, interpretable)

Additional factors:

Latency constraint:
├─ Real-time (<100ms): Use Hierarchical or Modular
└─ Planning can be slow (1-5s): Any architecture OK

Data availability:
├─ Abundant demos: End-to-End
├─ Few demos: Hierarchical or Modular
└─ No demos, use LLM only: Modular (code generation)

Task complexity:
├─ Single action per scene: Any architecture
├─ Multi-step reasoning: Hierarchical or Modular
└─ Complex physics: End-to-End (learns implicit)
```

---

## Detailed Design: Hierarchical Architecture (Recommended for Most Cases)

### Level 1: Semantic Planner (LLM-based)

```python
class SemanticPlanner:
    """LLM-based task decomposition and planning."""

    def __init__(self, model_name="llama2-70b"):
        self.llm = load_llm(model_name)
        self.vision_encoder = load_vision_encoder()

    def plan(self, image, task_description, scene_context=""):
        """Generate high-level plan from image and language."""

        # Encode image
        image_features = self.vision_encoder(image)
        image_description = self.generate_caption(image_features)

        # Prompt for planning
        prompt = f"""
        Scene: {image_description}
        {scene_context}

        Task: {task_description}

        Decompose this task into concrete sub-goals.
        For each sub-goal, specify:
        - Action (grasp, move, place, push, etc.)
        - Target object or location
        - Target position [x, y, z] in meters (robot base frame)
        - Parameters (gripper_width, force, speed)

        Output format (JSON):
        {{
          "sub_goals": [
            {{
              "action": "move_to_grasp",
              "target_object": "red cup",
              "target_position": [0.3, 0.2, 0.95],
              "gripper_width_m": 0.08,
              "approach_height_m": 0.15,
              "speed_ms": 0.5
            }},
            ...
          ],
          "estimated_success_rate": 0.8,
          "failure_modes": ["cup too close to edge", "gripper force too low"]
        }}
        """

        plan = self.llm(prompt)
        return json.loads(plan)
```

### Level 2: Reactive Control Policy (Neural Network)

```python
class ReactiveController:
    """Fast neural network for low-level control."""

    def __init__(self):
        # Lightweight network
        self.encoder = nn.Sequential(
            nn.Linear(14 + 3, 128),  # 7 joints + 7 velocities + 3D target
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.action_head = nn.Linear(128, 8)  # 7 joint vels + 1 gripper cmd

    def forward(self, state, subgoal):
        """
        Args:
            state: [joint_angles, joint_velocities] (14D)
            subgoal: [target_position] (3D)

        Returns:
            action: [joint_velocities, gripper_cmd] (8D)
        """

        # Compute error
        current_pos = self.compute_forward_kinematics(state[:7])
        error = subgoal - current_pos  # 3D error

        # Network prediction
        input_features = torch.cat([state, error], dim=-1)
        features = self.encoder(input_features)
        action = self.action_head(features)

        # Clamp to valid ranges
        joint_vels = torch.clamp(action[:7], -1.5, 1.5)  # rad/s
        gripper_cmd = torch.clamp(action[7], -1, 1)       # -1=close, +1=open

        return joint_vels, gripper_cmd
```

### Level 3: Execution Loop

```python
class VLAExecutor:
    """Orchestrate planner and controller."""

    def __init__(self):
        self.planner = SemanticPlanner()
        self.controller = ReactiveController()
        self.robot = RobotInterface()

    def execute(self, image, task):
        """Execute a task end-to-end."""

        # Step 1: Plan
        print(f"Planning: {task}")
        plan = self.planner.plan(image, task)

        # Step 2: Execute each sub-goal
        for i, subgoal in enumerate(plan['sub_goals']):
            print(f"Sub-goal {i}: {subgoal['action']}")
            self.execute_subgoal(subgoal)

        # Step 3: Verify success
        success = self.verify_success(task)
        return success

    def execute_subgoal(self, subgoal):
        """Execute a single sub-goal."""

        target_pos = np.array(subgoal['target_position'])
        max_iterations = 500  # ~10 seconds at 50Hz

        for t in range(max_iterations):
            # Get current state
            state = self.robot.get_state()  # [joint angles, velocities]

            # Get reactive action
            joint_vels, gripper_cmd = self.controller(state, target_pos)

            # Send to robot
            self.robot.command_joints(joint_vels)
            self.robot.command_gripper(gripper_cmd)

            # Check convergence
            current_pos = self.compute_forward_kinematics(state[:7])
            error = np.linalg.norm(current_pos - target_pos)

            if error < 0.02:  # 2cm tolerance
                print(f"Reached target (error={error:.3f}m)")
                break

            time.sleep(0.02)  # 50Hz control loop
```

---

## Training Strategy: How to Build Your Own VLA

### Option 1: From Pre-trained Models (Recommended)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig

# Start with pre-trained models
llm = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b")
vision_encoder = load_pretrained_vit("ViT-L/14")

# Add LoRA adapters (fast fine-tuning)
lora_config = LoraConfig(
    r=16,  # Low-rank dimension
    lora_alpha=32,  # Scaling
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

llm_lora = get_peft_model(llm, lora_config)

# Train only LoRA parameters (10% of model size)
optimizer = torch.optim.AdamW(llm_lora.parameters(), lr=1e-4)

# Fine-tune on your robot data
for epoch in range(10):
    for batch in robot_demonstrations:
        images, instructions, actions = batch

        # Forward pass
        predictions = llm_lora(images, instructions)
        loss = compute_loss(predictions, actions)

        # Backward pass (only LoRA parameters updated)
        loss.backward()
        optimizer.step()
```

**Cost & Time**:
- LoRA fine-tuning: 1-2 GPU days on single V100
- Data collection: 100-500 demonstrations (10-20 hours of robot)
- Total cost: ~$100 in compute, ~20 hours of effort

### Option 2: Custom Architecture

```python
class CustomVLAPolicy(nn.Module):
    """Build from scratch if you have specific requirements."""

    def __init__(self, robot_dof=7):
        super().__init__()

        # Vision backbone (freeze for faster training)
        self.vision_encoder = load_pretrained_vit()
        self.vision_encoder.requires_grad = False

        # Language backbone (freeze)
        self.language_encoder = load_pretrained_bert()
        self.language_encoder.requires_grad = False

        # Small trainable modules
        self.vision_projection = nn.Linear(768, 128)
        self.language_projection = nn.Linear(768, 128)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 + 128 + robot_dof, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.action_head = nn.Linear(128, robot_dof)

    def forward(self, image, language, joint_state):
        # Frozen encoders
        with torch.no_grad():
            vision_feat = self.vision_encoder(image)
            language_feat = self.language_encoder(language)

        # Project
        vision_proj = self.vision_projection(vision_feat)
        language_proj = self.language_projection(language_feat)

        # Fuse
        fused = torch.cat([vision_proj, language_proj, joint_state], dim=-1)
        hidden = self.fusion_mlp(fused)

        # Predict action
        action = self.action_head(hidden)

        return action
```

**Parameters**:
- Frozen backbones: 2B parameters (not trained)
- Trainable modules: 50K parameters (trained)
- Training time: 2-4 GPU hours on V100

---

## Real-World Examples

### Example 1: Google RT-2

```
Architecture: End-to-End
- Vision: ViT-B (86M params)
- Language: PaLM 2 (540B params, frozen)
- Fusion: Minimal (10M params)

Training:
- Data: 100K+ robot demonstrations
- Time: 4-8 weeks of GPU training
- Cost: $50K-100K in compute

Results:
- In-distribution success: 97%
- Zero-shot transfer: 76%
```

### Example 2: VoxPoser (CMU)

```
Architecture: Hierarchical
- LLM: GPT-4 (for planning)
- Vision: CLIP-ViT (for grounding)
- Control: Pre-trained diffusion policy

Training:
- Data: None (zero-shot with GPT-4 reasoning)
- Time: None (use API)
- Cost: ~$0.10 per task (API calls)

Results:
- Zero-shot success: 42-68%
- Single demonstration adaptation: 70-80%
```

### Example 3: Code-as-Policies (UC Berkeley)

```
Architecture: Modular (Code Generation)
- LLM: Codex (generates Python code)
- API: Pre-defined function library
  - find_object(color, shape)
  - move_to(position)
  - grasp(force)
  - place()

Training:
- Data: Function demonstrations only (not full trajectories)
- Time: 1-2 GPU days (learning individual skills)
- Cost: ~$1000 in compute

Results:
- Composable to novel tasks
- Success: 45-75% (depending on task complexity)
```

---

## Debugging Failed VLA Systems

### Common Failure Modes

```
Failure 1: Semantic Planning Wrong
├─ Symptom: "LLM predicts wrong target location"
├─ Root cause: Ambiguous scene or poor vision grounding
├─ Fix:
│  - Add scene description: "Table is at z=0.8m"
│  - Use multi-view: Multiple camera angles
│  - Verify: Check LLM output before executing

Failure 2: Grounding Error (Image → 3D)
├─ Symptom: "Robot moves to wrong place"
├─ Root cause: Camera calibration error or depth noise
├─ Fix:
│  - Recalibrate camera intrinsics
│  - Use depth filtering (bilateral filter)
│  - Verify: Compare visual and tactile feedback

Failure 3: Control Error
├─ Symptom: "Robot oscillates or misses target"
├─ Root cause: Control gains too high or low
├─ Fix:
│  - Tune PID gains (or use learned controller)
│  - Reduce target velocity
│  - Add damping

Failure 4: Physical Infeasibility
├─ Symptom: "IK solver fails, joint limits exceeded"
├─ Root cause: Target unreachable for this robot
├─ Fix:
│  - Add constraints: "Target must be within 0.6m"
│  - Clamp to workspace: project to nearest reachable
│  - Ask user: "Can you move the object closer?"
```

### Debugging Checklist

```python
def debug_vla_failure(task_description, failure_log):
    """Systematic debugging."""

    # Check 1: Is perception working?
    image = failure_log['camera_image']
    detections = detect_objects(image)
    assert len(detections) > 0, "No objects detected!"

    # Check 2: Is grounding working?
    world_pos = pixel_to_world(detections[0]['pixel_pos'], detections[0]['depth'])
    assert is_reachable(world_pos), f"Target {world_pos} out of reach!"

    # Check 3: Is planning working?
    plan = llm.plan(image, task_description)
    assert all_targets_reachable(plan), "Some targets infeasible!"

    # Check 4: Is control working?
    simulate_trajectory(plan)
    collisions = check_collisions()
    assert len(collisions) == 0, "Trajectory collides!"

    # If all checks pass, failure is likely hardware/calibration
    print("Systematic debugging complete. Likely causes:")
    print("- Calibration drift (recalibrate camera)")
    print("- Hardware slipping (check gripper friction)")
    print("- Model overfitting (collect more diverse data)")
```

---

## Key Takeaways

| Concept | Definition | When to Use |
|---------|-----------|---|
| **Hierarchical** | LLM planning + NN control | Most practical applications |
| **End-to-End** | Single network | When you have massive data |
| **Modular** | Code generation + APIs | Interpretable, composable tasks |
| **LoRA Fine-tuning** | Efficient adaptation | Fast training with limited data |
| **Skill Libraries** | Pre-learned primitives | Reliable, modular execution |
| **Reactive Control** | Low-level feedback loop | Real-time stability |

---

## Next Steps

1. **Implement a planner** → Study vla_policy_learner.py
2. **Evaluate your system** → Use vla_evaluation.py
3. **Debug failures** → Use exercises and challenge problems
4. **Deploy locally** → Set up Ollama and LM Studio

---

**Further Reading**:
- RT-2 Architecture: https://robotics-transformer-2.github.io/
- VoxPoser: https://arxiv.org/abs/2307.05973
- Code-as-Policies: https://code-as-policies.github.io/
