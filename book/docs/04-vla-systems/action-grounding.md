# Action Grounding: From Language to Motor Commands

The hardest part of embodied AI is closing the **semantic gap**: translating language concepts ("pick up") into continuous motor commands (7 joint angles + gripper force + timing). This chapter covers the techniques that make this translation work.

---

## The Semantic Gap Problem

### The Challenge

**Language Understanding** → "Pick up the red cup"

**Motor Control** → [θ₁=0.45, θ₂=0.32, θ₃=0.18, θ₄=-1.2, θ₅=0.8, θ₆=2.1, θ₇=0.05, gripper=0.08m, force=50N]

**The gap**: How do we get from the first to the second?

### Why It's Hard

1. **Abstraction mismatch**: Language is semantic ("pick up"), control is continuous (7D joint space)
2. **Many solutions**: Infinite ways to pick up a cup (top grasp, side grasp, two-handed, etc.)
3. **Context dependent**: Same word means different things (pick up ≠ pick up a spider)
4. **Real-time constraints**: Must generate actions fast (~100ms), not slowly reason (1-5s)

### Three Approaches

**Approach 1: Action Tokenization** (Discrete actions)
```
Language: "Pick up"
  ↓
LLM outputs: "ACTION_GRASP_FROM_TOP"
  ↓
Decode to motor commands (pre-defined trajectories)
```

**Approach 2: Action Regression** (Continuous prediction)
```
Language: "Pick up"
  ↓
LLM outputs: "target_position=[0.3, 0.2, 0.8], gripper_width=0.08, force=50"
  ↓
Neural network converts to joint angles via inverse kinematics
```

**Approach 3: Diffusion-Based** (Iterative refinement)
```
Language: "Pick up"
  ↓
Start with random action noise
  ↓
Iteratively refine using language guidance + physics simulation
  ↓
Final action predicted
```

---

## Approach 1: Action Tokenization

### Idea

Treat actions like words in language. Instead of 7 continuous joint values, learn a finite vocabulary of discrete actions.

### Example Vocabulary

```python
ACTION_TOKENS = {
    # Grasping actions
    0: "APPROACH_FROM_TOP",           # Reach object from above
    1: "APPROACH_FROM_SIDE",          # Reach from side
    2: "APPROACH_FROM_FRONT",         # Reach from front
    3: "GRASP_AND_LIFT",              # Close gripper and lift
    4: "GRASP_GENTLE",                # Close gripper slowly (fragile)

    # Movement actions
    5: "MOVE_TO_POSITION_FAST",       # 0.8 m/s
    6: "MOVE_TO_POSITION_SLOW",       # 0.2 m/s (precision)
    7: "ROTATE_GRIPPER_CW",           # Rotate 15°
    8: "ROTATE_GRIPPER_CCW",          # Rotate -15°

    # Release actions
    9: "RELEASE_NORMAL",              # Open gripper (normal speed)
    10: "RELEASE_GENTLE",             # Open gripper slowly

    # Safety
    11: "RETREAT_VERTICAL",           # Pull straight up
    12: "STOP_EMERGENCY",             # Stop and hold position
}
```

### How It Works

```python
class ActionTokenizerPolicy(nn.Module):
    """Predict discrete action tokens given language + vision."""

    def __init__(self):
        # Vision encoder
        self.vision_encoder = load_pretrained_vit()  # [B, 2048]

        # Language encoder
        self.language_encoder = load_pretrained_bert()  # [B, 768]

        # Fusion and action head
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.action_head = nn.Linear(256, num_action_tokens)  # 13 tokens

    def forward(self, image, language):
        """
        Args:
            image: [B, 3, 224, 224]
            language: [B, seq_len] token IDs

        Returns:
            action_logits: [B, num_actions]
            action_token: [B] predicted action token
        """
        # Encode
        vision_feat = self.vision_encoder(image)  # [B, 2048]
        language_feat = self.language_encoder(language)  # [B, 768]

        # Fuse
        fused = torch.cat([vision_feat, language_feat], dim=1)  # [B, 2816]
        hidden = self.fusion(fused)  # [B, 256]

        # Predict action
        action_logits = self.action_head(hidden)  # [B, 13]
        action_token = torch.argmax(action_logits, dim=1)  # [B]

        return action_logits, action_token

def decode_action_token(action_token, object_position, robot_state):
    """Convert action token to motor command."""

    if action_token == 0:  # APPROACH_FROM_TOP
        # Plan trajectory from current position to above object
        target = object_position + [0, 0, 0.15]  # 15cm above
        trajectory = plan_trajectory(robot_state, target, approach_height=0.5m/s)

    elif action_token == 3:  # GRASP_AND_LIFT
        # Close gripper (width → 0), then lift 0.3m
        gripper_trajectory = close_gripper_trajectory(speed=0.1)  # Slow
        lift_trajectory = lift_trajectory(height=0.3, speed=0.2m/s)
        trajectory = gripper_trajectory + lift_trajectory

    # ... more cases ...

    return trajectory
```

### Advantages & Disadvantages

| Advantage | Disadvantage |
|-----------|--------------|
| Fast inference (single token prediction) | Limited expressiveness (finite actions) |
| Interpretable (can see what LLM chose) | Doesn't adapt to novel situations |
| Easy to constrain (only allow safe tokens) | Requires pre-defining all actions |
| Works with small models (Phi 3.8B) | Struggles with continuous variations |

**Best for**: Repetitive, well-defined tasks (pick-and-place, assembly lines)

---

## Approach 2: Action Regression

### Idea

LLM outputs semantic parameters (target position, gripper force), then neural network translates to joint angles.

### Architecture

```python
class ActionRegressionPolicy(nn.Module):
    """Predict continuous action parameters."""

    def __init__(self, robot_dof=7):
        # Vision + Language backbone (shared)
        self.vision_encoder = load_pretrained_vit()    # 2048-dim
        self.language_encoder = load_pretrained_bert()  # 768-dim

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 768 + robot_dof, 512),  # Add current joint angles
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Output semantic action parameters
        self.semantic_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7),  # [x, y, z, roll, pitch, yaw, gripper_width]
        )

        # Inverse kinematics network (learned or analytical)
        self.ik_network = LearnedIK(dof=robot_dof)

    def forward(self, image, language, joint_state):
        """
        Args:
            image: [B, 3, 224, 224]
            language: [B, seq_len]
            joint_state: [B, 7]

        Returns:
            joint_trajectory: [B, T, 7] (T timesteps)
        """
        # Encode
        vision_feat = self.vision_encoder(image)
        language_feat = self.language_encoder(language)

        # Fuse with proprioception
        fused_input = torch.cat(
            [vision_feat, language_feat, joint_state],
            dim=1
        )
        hidden = self.fusion(fused_input)

        # Predict semantic action
        semantic_action = self.semantic_head(hidden)  # [B, 7]
        # [target_x, target_y, target_z, roll, pitch, yaw, gripper_width]

        # Convert to joint angles via IK
        joint_solution = self.ik_network(semantic_action)  # [B, 7]

        # Generate trajectory (e.g., linear interpolation)
        current_joints = joint_state  # [B, 7]
        trajectory = interpolate_trajectory(
            start=current_joints,
            end=joint_solution,
            num_steps=50  # 50 timesteps (2.5s at 20Hz)
        )

        return trajectory  # [B, 50, 7]
```

### Training

```python
def train_action_regression(model, dataset):
    """Train on robot demonstrations."""

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(100):
        for batch in dataset:
            images = batch['images']              # [B, 3, 224, 224]
            languages = batch['instructions']     # [B, seq_len]
            joint_states = batch['joint_states']  # [B, 7]
            expert_actions = batch['actions']     # [B, 7] (target positions/forces)

            # Forward
            predicted_trajectory = model(images, languages, joint_states)
            last_prediction = predicted_trajectory[:, -1, :]  # Final action [B, 7]

            # Loss: L1 distance between predicted and expert
            loss = F.l1_loss(last_prediction, expert_actions)

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model
```

### Advantages & Disadvantages

| Advantage | Disadvantage |
|-----------|--------------|
| Continuous, adaptive output | More complex (requires IK solver) |
| Generalizes to novel objects | Slower (multiple networks in series) |
| Outputs interpretable parameters | IK errors propagate to final action |

**Best for**: Flexible manipulation tasks (grasp any object, place anywhere)

---

## Approach 3: Diffusion-Based Actions

### Idea

Use diffusion models (like DALL-E) but for robot actions instead of images.

**Diffusion Process**:
```
Step 1: Start with random action noise
        a₀ ~ N(0, I)

Step 2: Iteratively refine using:
        a_t = a_{t-1} - λ * ∇_a L(a, instruction, image)

Step 3: After K iterations, a_K is the final action

This is like asking "what action would a good robot take?"
and gradually refining the answer.
```

### Architecture

```python
class DiffusionActionPolicy(nn.Module):
    """Learn to generate actions via diffusion process."""

    def __init__(self, num_diffusion_steps=20):
        self.num_steps = num_diffusion_steps

        # Noise prediction network (U-Net style)
        self.noise_predictor = nn.Sequential(
            nn.Linear(7 + 2048 + 768 + num_diffusion_steps, 256),  # action + vision + lang + timestep
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 7),  # Predict noise in action space
        )

    def forward(self, image, language, num_steps=20):
        """
        Args:
            image: [B, 3, 224, 224]
            language: [B, seq_len]
            num_steps: Number of diffusion iterations

        Returns:
            action: [B, 7] final denoised action
        """

        # Encode
        vision_feat = self.encode_vision(image)   # [B, 2048]
        language_feat = self.encode_language(language)  # [B, 768]

        # Initialize with noise
        action = torch.randn(image.shape[0], 7)  # [B, 7]

        # Iterative denoising
        for t in range(num_steps):
            # Concatenate with noise level encoding
            t_embedding = self.encode_timestep(t, num_steps)  # [B, num_steps]

            # Predict noise
            noise_pred = self.noise_predictor(
                torch.cat([action, vision_feat, language_feat, t_embedding], dim=1)
            )

            # Update action (remove predicted noise)
            alpha = 0.1  # Step size
            action = action - alpha * noise_pred

            # Add small random noise for exploration (optional)
            if t < num_steps - 1:
                action += 0.01 * torch.randn_like(action)

        return action
```

### Advantages & Disadvantages

| Advantage | Disadvantage |
|-----------|--------------|
| Flexible and powerful | Slow (K iterations of inference) |
| Can model multimodal actions | Complex to train |
| Handles uncertainty naturally | Requires large dataset |

**Best for**: Complex manipulation tasks with multiple valid solutions

---

## Comparison of Approaches

| Method | Speed | Generalization | Expressiveness | Training Data |
|--------|-------|---|---|---|
| **Tokenization** | Fastest (1 forward pass) | Low (only learned tokens) | Limited | Smallest |
| **Regression** | Medium (1-2 forward passes) | Medium-High | Moderate | Medium |
| **Diffusion** | Slowest (K forward passes) | High | High | Largest |

**Recommendation**:
- Start with **tokenization** if tasks are repetitive
- Use **regression** for flexible manipulation
- Use **diffusion** if you have data and need best results

---

## Beyond Raw Actions: Skill Libraries

### Idea

Instead of predicting low-level joint angles, predict **high-level skills** that are pre-learned.

```python
SKILL_LIBRARY = {
    "grasp_from_top": GraspSkill(approach_height=0.15, force=50),
    "grasp_from_side": GraspSkill(approach_height=0.0, force=50),
    "push": PushSkill(distance=0.2, force=30),
    "place_gently": PlaceSkill(descent_speed=0.1, force=20),
    "open_drawer": DrawerSkill(pull_distance=0.3),
}

class SkillBasedPolicy:
    def forward(self, image, language):
        # LLM predicts which skill to use
        skill_name = self.llm(image, language)  # "grasp_from_top"

        # Get skill and object parameters
        skill = SKILL_LIBRARY[skill_name]
        target = self.detect_object(image)

        # Execute skill with target
        trajectory = skill(target_position=target)

        return trajectory
```

**Advantages**:
- Modular and reusable
- Each skill is proven to work
- LLM only needs to choose skill (simpler)
- Easy to add new skills

**Disadvantages**:
- Requires manual skill engineering
- Doesn't handle novel situations
- Skills must have clear semantics

---

## End-to-End Learning: The Alternative

### Raw Neural Network Policy

```python
class EndToEndPolicy(nn.Module):
    """Single neural network: Image + Language → Motor Commands"""

    def __init__(self):
        self.vision_encoder = ResNet50(pretrained=True)
        self.language_encoder = BERT.from_pretrained("bert-base")

        self.fusion = nn.Sequential(
            nn.Linear(2048 + 768 + 7, 512),  # vision + language + proprioception
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Direct output: 7 joint angles + gripper width + 50 timesteps
        self.action_head = nn.Linear(256, (7 + 1) * 50)

    def forward(self, image, language, joint_state):
        vision = self.vision_encoder(image)
        language = self.language_encoder(language)
        fused = self.fusion(torch.cat([vision, language, joint_state], dim=1))

        # Output 50-step trajectory
        action_sequence = self.action_head(fused)
        action_sequence = action_sequence.reshape(-1, 50, 8)  # [B, 50, 8]

        return action_sequence
```

**Pros**: No semantic bottleneck, learns implicit representations
**Cons**: Black box, requires massive training data (millions of demos)

---

## Key Takeaways

| Approach | Best For | Key Insight |
|----------|----------|---|
| **Tokenization** | Discrete, repetitive tasks | Simplify action space |
| **Regression** | Flexible manipulation | Use intermediate representations |
| **Diffusion** | Complex, multi-solution tasks | Iteratively refine actions |
| **Skill Libraries** | Modular, reusable behaviors | Combine learned primitives |
| **End-to-End** | When you have massive data | Learn everything jointly |

---

## Next Steps

1. **See system architecture** → Read vla-architecture.md
2. **Implement it** → Study vla_policy_learner.py
3. **Evaluate performance** → Use vla_evaluation.py
4. **Apply to real task** → Complete exercises

---

**Further Reading**:
- RT-1 (Action Tokens): https://robotics-transformer.github.io/
- Diffusion Policies: https://diffusion-policy.cs.columbia.edu/
- Skill Discovery: https://arxiv.org/abs/2304.04435
