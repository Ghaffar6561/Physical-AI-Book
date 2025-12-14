# Vision-Language Models for Robotics

This chapter explains how large language models and vision-language models can be adapted for robotic control. You'll learn the fundamental architectures, why they work, and their limitations for embodied tasks.

---

## Foundation Models: The New Paradigm

### What Changed in 2022-2023?

For decades, AI systems required task-specific training. Want to classify images? Train an ImageNet model. Want to translate languages? Train a separate translation model. Want to control robots? Write explicit C++ code.

**Foundation Models** changed this. A single model trained on massive internet-scale data can:
- Read images from any domain
- Understand text in any language
- Follow instructions it's never seen
- Transfer knowledge to completely new tasks

**Timeline of breakthroughs:**
- **2017**: Transformer architecture introduced (Vaswani et al., "Attention is All You Need")
- **2020**: GPT-3 shows in-context learning (~100B parameters)
- **2021**: CLIP shows vision-language co-training works at scale (400M image-text pairs)
- **2022**: ChatGPT reaches 100M users; shows that LLM instruction-following transfers to robotics
- **2023**: GPT-4, Claude, Llama 2 show even better reasoning; RT-2 shows end-to-end VLA works with few demos
- **2024**: Open models (Llama 3, Mistral) reach parity with closed models; local deployment becomes practical

### Why Foundation Models Work for Robotics

Foundation models succeed because they capture **general world knowledge**:

```
"Pick up the cup" is decomposed by the model as:
├── Object recognition (cup = cylindrical container)
├── Physics understanding (cups are fragile, need gentle grip)
├── Spatial reasoning (cup location in image & world frame)
├── Motor planning (grasp approach, force control)
└── Safety awareness (don't spill liquid, don't drop fragile object)
```

This knowledge comes from training on:
- Billions of images with captions (ImageNet, LAION, YFCC100M)
- Trillions of words of text (Common Crawl, Wikipedia, books)
- Real robot demonstrations (ALOHA dataset, BridgeData, RoboPilot)

A robot doesn't need to learn "cup" from scratch—it already knows from training.

---

## The Foundation Model Taxonomy

### Large Language Models (LLMs)

**Definition**: Neural networks trained on text to predict next tokens.

**Architecture**:
```
Input: "Pick up the red cube on the..."
        ↓
Token embeddings: [Pick=0.2, up=0.1, the=0.05, ...]
        ↓
Transformer layers (12-96 layers, each with ~8 attention heads)
        ↓
Predict next token: "left" (confidence=0.85), "right" (confidence=0.10), ...
        ↓
Output: [4, 1, 8, 7, ...] (token IDs for next words)
```

**Popular Models**:
- **Closed (API-based)**:
  - GPT-4 (OpenAI): 1.7T parameters, $20/million input tokens
  - Claude 3 Opus (Anthropic): ~100B parameters, $15/million tokens
  - Gemini (Google): ~170B parameters (not all shown)

- **Open (Self-hosted)**:
  - Llama 2 (Meta): 7B-70B parameters, MIT license
  - Mistral (Mistral AI): 7B-8x22B MoE parameters, Apache 2.0
  - Phi (Microsoft): 3.8B parameters, fast inference

**For Robotics**: Use **reasoning-capable models** (Claude, GPT-4) for planning; use **fast models** (Phi, Mistral 7B) for low-latency inference.

### Vision Transformers

**Definition**: Neural networks trained on images to predict visual features.

**Architecture**:
```
Input image (224×224×3 RGB)
     ↓
Split into 16×16 patches (196 patches)
     ↓
Embed each patch (→ 768-dim vectors)
     ↓
Transformer encoder (12-24 layers)
     ↓
Output: 196 patch embeddings (visual features)
```

**Popular Models**:
- **ViT** (Vision Transformer): 86M-303M parameters, trained on ImageNet
- **DeiT** (Distilled ViT): Smaller, faster (66M parameters)
- **TIMM**: PyTorch Image Models with 400+ pretrained vision models

**For Robotics**: Use **pretrained vision backbones** to extract image features. Then fine-tune only the action head (2-5% of parameters).

### Vision-Language Models (VLMs)

**Definition**: Single neural network that understands both images and text.

**Architecture** (Example: CLIP):
```
Input: (Image, Text)
        ↓
        ├─→ Vision Encoder (ViT)     → Image embedding (512-dim)
        │
        └─→ Text Encoder (Transformer) → Text embedding (512-dim)

        ↓ (during training)
        Contrastive loss: max similarity between matching pairs
```

**How It Works**:
1. Train on 400M image-caption pairs
2. Learn to map images and text to a shared embedding space
3. At test time, encode image and text separately; similarity = dot product

**Example**:
```python
from clip import load_model_and_processor

model, processor = load_model_and_processor("ViT-L/14")

# Image
image = Image.open("cup.jpg")
image_emb = model.encode_image(processor(image))  # [512-dim]

# Text (candidate descriptions)
texts = ["a cup", "a bowl", "a bottle", "a mug"]
text_embs = model.encode_text(processor(texts))   # [4, 512-dim]

# Which description matches the image?
similarities = image_emb @ text_embs.T           # [4]
best_match = texts[np.argmax(similarities)]     # "a cup" or "a mug"
```

**Popular Models**:
- **CLIP** (OpenAI): 63M-336M parameters, trained on 400M image-text pairs
- **BLIP** (Salesforce): 225M parameters, includes generation (can caption images)
- **LLaVA** (UMass): Llama + CLIP vision encoder, instruction-tuned
- **Qwen-VL** (Alibaba): Vision transformer + Qwen language model, multilingual

**For Robotics**: Use **multimodal embeddings** to ground language in images. Enables queries like:
- "Find the red cup in this image" (visual grounding)
- "Describe what the robot should do" (image captioning)
- "Is the gripper approaching correctly?" (visual question answering)

---

## How Foundation Models Work for Robot Control

### The Three-Step Process

**Step 1: Encode Multimodal Input**
```python
# Combine vision + language
image_embedding = vision_encoder(camera_image)           # [B, 2048]
language_embedding = text_encoder("Pick up the cup")     # [512]
proprioception = [joint_angles, gripper_position]        # [14]

# Concatenate
input_features = concatenate([image_embedding, language_embedding, proprioception])
# [B, 2048 + 512 + 14] = [B, 2574]
```

**Step 2: Foundation Model Reasoning**
```python
# Pass through LLM (frozen or fine-tuned)
reasoning = llm(input_features, task_description)

# Example output:
# "The red cup is located at (x=0.3m, y=0.2m, z=0.8m).
#  Approach from above at 0.5 m/s with gripper width 0.08m.
#  Grasp force: 50N. Lift 0.3m vertically."
```

**Step 3: Action Grounding**
```python
# Convert reasoning to motor commands
parsed_action = parse_reasoning(reasoning)

# {
#   "target_position": [0.3, 0.2, 0.8],
#   "approach_velocity": 0.5,
#   "gripper_width": 0.08,
#   "grasp_force": 50,
#   "lift_height": 0.3
# }

# Inverse kinematics
joint_trajectory = compute_ik(parsed_action)
send_to_robot(joint_trajectory)
```

### Concrete Example: RT-2 (Google)

**RT-2** (Robotics Transformer 2, 2023) shows how to make this work at scale:

**Architecture**:
```
Camera image (480×640×3)
    ↓
Vision backbone (ViT): Extract 1024-dim features
    ↓
Language model (PaLM 2): Reason about task
    ↓
Decode action tokens: [action_x, action_y, action_z, gripper, terminate]
    ↓
Motor controller: Send to robot
```

**Training**:
- Use **reward** from picking up objects (did gripper close on object?)
- Fine-tune on multi-task demonstrations (pick, place, push, slide)
- No explicit action supervision needed; learn from outcomes

**Results**:
- 97% success on in-distribution tasks
- 76% success on novel tasks (zero-shot transfer)
- Learns from 100K demonstrations (takes weeks of robot learning)

**Why it works**:
1. Vision encoder already understands objects from ImageNet training
2. Language model already understands goals from text training
3. Action head is small (5-10K parameters) and learns quickly
4. Reward signal is sparse but clear (success/failure)

---

## Core Capabilities for Robotics

### Capability 1: Semantic Understanding

LLMs understand meaning without explicit supervision:

```python
instructions = [
    "Pick up the red cube",
    "Grasp the red block",
    "Take the crimson stone",
]

# All three understood as similar task
# Despite different words, same spatial semantics
```

**Why it matters**: Robot can follow open-ended natural language instructions. No need to pre-program every possible command.

### Capability 2: Contextual Reasoning

LLMs maintain context across conversation:

```
User: "Stack the cube on the red block. But avoid the bottle."
LLM reasons:
├─ Primary task: Stack red cube on red block
├─ Constraint: Don't knock over the bottle
├─ Sub-goal 1: Locate red cube and red block
├─ Sub-goal 2: Plan collision-free path (avoid bottle)
└─ Sub-goal 3: Execute grasp, move, place, release
```

**Why it matters**: Robot can handle multi-step tasks and constraints. Learns task decomposition from experience.

### Capability 3: Few-Shot Learning

LLMs learn from examples:

```python
# Zero-shot (no examples)
response = llm("Pick up the cup")  # Works ~30% of time

# Few-shot (2-3 examples)
prompt = """
Example 1:
Image: [cup on table]
Task: "Pick up the cup"
Action: [gripper_x=0.5, gripper_y=0.2, force=50, lift_height=0.3]

Example 2:
Image: [bottle on table]
Task: "Pick up the bottle"
Action: [gripper_x=0.6, gripper_y=0.1, force=45, lift_height=0.25]

Now perform: Pick up the cup
Action: [?]
"""

response = llm(prompt)  # Works ~70% of time
```

**Why it matters**: Robot can adapt to new tasks and objects quickly. No need to retrain from scratch.

### Capability 4: Chain-of-Thought Reasoning

LLMs can articulate their reasoning:

```python
response = llm("""
Image shows a cup at position (0.3, 0.2, 0.8).
Task: "Pick up the cup and place it on the shelf at (0.5, 0.5, 1.2)".

Let me think step by step:
1. The cup is currently at (0.3, 0.2, 0.8). It's about 15cm tall and 8cm wide.
2. I need to approach from above to avoid tipping it.
3. My gripper width should be ~6cm to grasp without damaging.
4. After grasping, I lift 0.3m and move horizontally to shelf position (0.5, 0.5).
5. Finally, I place it gently on the shelf at height 1.2m.

Actions:
- Approach: position=(0.3, 0.2, 1.0), orientation=(0, 0, 1)
- Grasp: force=50N, gripper_width=0.06
- Lift: height_delta=0.3
- Move: target=(0.5, 0.5, 1.2)
- Release: force=0
""")

# Structured reasoning → better accuracy
```

**Why it matters**: Robot can explain its decisions. Useful for debugging and human oversight.

---

## Limitations & Failure Modes

### Limitation 1: Hallucination

**Problem**: LLMs confidently generate false information.

**Example**:
```
Instruction: "Place the cube in the red drawer"
LLM reasoning: "I see the red drawer is on the left side."
Reality: There is no drawer. The red box is a solid container.
Result: Robot attempts impossible action, fails.
```

**Mitigation**:
- Add vision grounding: "The image shows: [detected objects]"
- Ask for confidence: "How confident are you? (0-100%)"
- Require verification: Check success before next step

### Limitation 2: Real-Time Latency

**Problem**: Large LLMs take 1-5 seconds per inference. Robots need ~100ms.

**Example**:
```
Camera FPS: 30 (33ms per frame)
LLM reasoning time: 2000ms
By the time LLM decides, the object has moved 2 frames (66ms late)
Robot reaches stale position, grasp fails.
```

**Mitigation**:
- Use hierarchical planning: LLM plans goals (2s OK), neural network controls (20ms)
- Compress models: Quantization (8-bit), distillation, pruning
- Cache decisions: Reuse reasoning for repeated tasks

### Limitation 3: Embodiment Gap

**Problem**: LLM trained on text about robots doesn't understand actual kinematics.

**Example**:
```
Instruction: "Reach the object on the high shelf"
LLM output: "Extend arm to (0.5, 0.5, 2.5)"
Reality: Robot has 0.6m reach. Maximum height is 1.8m.
Result: Out-of-workspace error.
```

**Mitigation**:
- Explicit constraints: "The robot has 0.6m reach. Max height is 1.8m."
- Action normalization: Convert to robot's native coordinate frame
- Empirical calibration: Learn mapping between LLM and motor commands

### Limitation 4: Safety

**Problem**: LLMs don't understand real-world safety constraints.

**Example**:
```
Task: "Pick up the object from the shelf"
LLM output: Move at 1.0 m/s with maximum force
Reality: Could hit human, knock over fragile items, exceed joint limits
Result: Safety violation.
```

**Mitigation**:
- Constraint checker: Verify actions before sending to robot
- Safety layer: Velocity limits, collision checking, torque limits
- Human-in-the-loop: For safety-critical tasks, require approval

---

## Practical Comparison: GPT-4 vs Llama 2 vs Local Models

| Model | Parameters | API Cost | Latency | Reasoning | Best For |
|-------|-----------|----------|---------|-----------|----------|
| **GPT-4** | ~1.7T (est.) | $15/1M tokens | 1-3s | Excellent (99%) | Planning, few-shot learning |
| **Claude 3 Opus** | ~100B | $15/1M tokens | 1-3s | Excellent (95%) | Long-context, nuanced reasoning |
| **Llama 2 70B** | 70B | Free (self-hosted) | 2-5s | Good (85%) | Open-source, no API calls |
| **Mistral 7B** | 7B | Free (self-hosted) | 100ms | Decent (70%) | Fast local inference |
| **Phi 3.8B** | 3.8B | Free (self-hosted) | 50ms | Fair (60%) | Edge devices, real-time control |

**Recommendation**:
- **High accuracy**: Use GPT-4 for planning (slow OK)
- **Real-time control**: Use Llama 7B or smaller (quantized)
- **On-device**: Use Phi or Mistral 7B (1-5GB VRAM)

---

## Vision-Language Models Compared

| Model | Vision Encoder | Language Model | Use Case | Strength |
|-------|---|---|---|---|
| **CLIP** | ViT (336M) | Text Encoder (63M) | Image classification, grounding | Fast inference (100ms) |
| **BLIP** | ViT (225M) | Captions generator | Image captioning, VQA | Generation capability |
| **LLaVA** | CLIP-ViT + Llama | Llama 2 (7B-13B) | Task reasoning, planning | Instruction-tuned for language |
| **Qwen-VL** | ViT + QwenLM | Qwen (7B-32B) | Multilingual, document understanding | Handles text in images |

**For robotics**, **LLaVA** is often the sweet spot:
- Can reason about robot tasks (Llama language understanding)
- Can ground in images (CLIP vision encoding)
- Instruction-tuned (follows natural language)
- Available open-source (no API calls)

---

## How to Use Foundation Models in Your Robot

### Option 1: Zero-Shot Prompting (Simplest, Lowest Cost)

```python
import openai

def robot_planner(image_path, task_description):
    """Use GPT-4 to plan robot task (zero-shot)."""

    # Encode image
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    # Prompt GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Task: {task_description}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content

# Usage
plan = robot_planner("camera.jpg", "Pick up the red cube")
print(plan)
# Output: "I see a red cube at position (0.3, 0.2, 0.8)..."
```

**Pros**: Simple, no training, works immediately
**Cons**: Slower (~3s/inference), API cost (~$0.03 per image), less reliable (<40% success on novel tasks)

### Option 2: Few-Shot Fine-Tuning (Balanced)

```python
from lora import LoRA
import torch

def fine_tune_lora(model, train_data, epochs=10):
    """Fine-tune LLaVA with LoRA (low-rank adaptation)."""

    # Freeze vision encoder, add LoRA to language model
    lora_config = LoRA(
        r=8,              # Low-rank dimension
        lora_alpha=16,    # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    )

    model = lora_config.get_peft_model(model)

    # Fine-tune on your robot data
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for batch in train_data:
            images, tasks, actions = batch

            output = model(images, tasks)
            loss = compute_action_loss(output, actions)

            loss.backward()
            optimizer.step()

    return model

# Usage
model = load_pretrained_llava("llava-7b")
fine_tuned_model = fine_tune_lora(model, robot_demonstrations)
fine_tuned_model.save("robot_vla_7b_lora.pth")
```

**Pros**: Moderate accuracy (~70%), fast inference (<200ms), cheap training (1-2 GPU days)
**Cons**: Requires ~100-500 demonstrations

### Option 3: Full Fine-Tuning (Best Accuracy)

```python
def full_finetune(model, train_data, num_epochs=5):
    """Full fine-tuning (more expensive, better results)."""

    # All parameters are trainable
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    for epoch in range(num_epochs):
        for batch in train_data:
            images, tasks, actions = batch

            # Forward pass
            output = model(images, tasks)
            loss = compute_action_loss(output, actions)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

    return model

# Usage (requires substantial compute)
model = load_pretrained_llava("llava-13b")
fully_trained = full_finetune(model, large_robot_dataset)  # 1000s of demos
fully_trained.save("robot_vla_13b_full.pth")
```

**Pros**: Highest accuracy (~90%), learned task distributions
**Cons**: Expensive (1-2 weeks of GPU training), requires large dataset (1000+ demos)

---

## Key Takeaways

| Concept | Meaning | Why It Matters |
|---------|---------|---|
| **Foundation Models** | Pre-trained on billions of examples | Transfer to robotics with little task-specific data |
| **Multimodal** | Understand vision + language together | Can ground language in camera images |
| **In-Context Learning** | Learn from examples in the prompt | Few-shot adaptation without retraining |
| **Vision-Language Alignment** | Images & text mapped to shared space | Enable visual grounding of language concepts |
| **Action Grounding** | Convert semantic concepts to motor commands | Bridge gap between planning and control |
| **Hierarchical Planning** | Separate reasoning from control | Solve real-time latency problem |

---

## Next Steps

1. **Understand spatial reasoning** → Read embodied-reasoning.md
2. **Learn action grounding** → Read action-grounding.md
3. **See architectural patterns** → Read vla-architecture.md
4. **Code it up** → Use vla_policy_learner.py to fine-tune
5. **Deploy locally** → Set up Ollama, run inference on your robot

---

**Further Reading**:
- CLIP (Radford et al., 2021): https://arxiv.org/abs/2103.00020
- RT-2 (Brohan et al., 2023): https://robotics-transformer-2.github.io/
- LLaVA (Liu et al., 2023): https://github.com/haotian-liu/LLaVA
- Llama 2 (Touvron et al., 2023): https://arxiv.org/abs/2307.09288
