# Module 4: VLA Systems & Embodied AI - Exercises

These exercises guide you through applying vision-language-action concepts to real robotics challenges. Work through them in order, from simple zero-shot prompting to implementing full VLA systems.

---

## Exercise 1: Zero-Shot VLA with GPT-4

### Objective

Understand how language models can control robots without any fine-tuning. Use GPT-4 Vision API to analyze a robot scene and generate action commands.

### Scenario

You have:
- A robot arm with 7 joints + gripper
- A camera feed showing a table with objects (cube, bottle, ball)
- Access to GPT-4 Vision API
- 5 minutes to implement basic VLA

### Task

1. **Take a screenshot** of a scene (real or from simulation)
2. **Write a prompt** asking GPT-4 to analyze the scene and suggest a robot action
3. **Parse the response** into executable motor commands
4. **Execute and evaluate** (did it make sense?)

### Success Criteria

✅ GPT-4 correctly identifies objects in the scene
✅ GPT-4 suggests a reasonable action (e.g., "grasp the red cube")
✅ You convert the response to [target_x, target_y, gripper_width, force]
✅ Zero-shot success rate ≥ 30%

### Solution

**Key insight**: Language models understand spatial concepts from training on internet text. They can reason about robot actions even without seeing robot-specific data.

**Success rates**:
- Zero-shot (no examples): ~30-40%
- Why it works: Models have seen "pick up" language and scene descriptions
- Why it fails: Doesn't understand robot kinematics (what's reachable)

---

## Exercise 2: Few-Shot Learning with LLaVA

### Objective

Improve accuracy by showing the LLM examples. Implement few-shot prompting with LLaVA (open-source vision-language model).

### Scenario

You have:
- A local LLaVA model (can run on laptop)
- 10 labeled robot demonstrations (image + action pairs)
- Need to boost performance from 30% → 70%

### Task

1. **Install LLaVA** and download the model (7B version, ~4GB)
2. **Create few-shot examples** from your 10 demonstrations
3. **Implement in-context learning** by including examples in prompt
4. **Test on held-out examples** and measure accuracy improvement

### Success Criteria

✅ LLaVA installed and running locally
✅ You can inference in <2 seconds per image
✅ Few-shot examples reduce hallucinations
✅ Success rate improves to 60-75%

### Solution

**Few-shot template**:
```
Example 1: [image] → Action: target=[x1,y1,z1], force=50N
Example 2: [image] → Action: target=[x2,y2,z2], force=45N
...
Now perform: [new image] → Action: ?
```

**Key insight**: In-context examples teach the model the task distribution without retraining.

**Success rates**:
- Few-shot (2-5 examples): ~60-75%
- Why it improves: Model learns to output format + parameter ranges
- Limitation: Doesn't generalize beyond example distribution

---

## Exercise 3: Fine-Tuning a VLA Policy

### Objective

Implement the full VLA pipeline: collect data, fine-tune a model, and deploy.

### Scenario

You have:
- 200 robot demonstrations (image + instruction + action)
- Pre-trained CLIP vision encoder + BERT language encoder
- Need 80%+ success rate on your specific task

### Task

1. **Organize data** as RobotDemonstration objects
2. **Create DataLoader** (use RobotDemonstrationDataset from vla_policy_learner.py)
3. **Fine-tune** only the action head (freeze vision/language encoders)
4. **Evaluate** on held-out test set
5. **Compare** zero-shot vs few-shot vs fine-tuned accuracy

### Success Criteria

✅ Data organized in standard format
✅ Training loop runs without errors
✅ Loss decreases over epochs
✅ Fine-tuned model >80% success on test set
✅ Shows improvement over zero-shot

### Solution Template

```python
from vla_policy_learner import VLAPolicyLearner, RobotDemonstration, RobotDemonstrationDataset
import torch
from torch.utils.data import DataLoader

# 1. Load demonstrations
demonstrations = [
    RobotDemonstration(
        image=load_image(f"demo_{i}.png"),
        instruction=load_instruction(f"demo_{i}.json"),
        target_position=load_target(f"demo_{i}.json"),
        gripper_width=0.08,
        grasp_force=50,
        approach_height=0.15,
        success=True,
    )
    for i in range(200)
]

# 2. Split and create dataloaders
split = int(0.8 * len(demonstrations))
train_loader = DataLoader(
    RobotDemonstrationDataset(demonstrations[:split]),
    batch_size=16, shuffle=True
)
test_loader = DataLoader(
    RobotDemonstrationDataset(demonstrations[split:]),
    batch_size=32
)

# 3. Initialize policy
policy = VLAPolicyLearner().to("cuda")
optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=1e-4)

# 4. Train
for epoch in range(10):
    losses = policy.train_epoch(train_loader, optimizer, "cuda")
    metrics = policy.evaluate(test_loader, "cuda")
    print(f"Epoch {epoch}: Loss={losses['total_loss']:.4f}")

# 5. Evaluate all approaches
for approach in ['zero_shot', 'few_shot', 'fine_tuned']:
    success_rate = evaluate_on_test_set(approach)
    print(f"{approach}: {success_rate*100:.1f}%")
```

**Expected result**:
- Zero-shot: 30-40%
- Few-shot: 60-70%
- Fine-tuned: 80-90%

**Key insight**: Fine-tuning adapts the action head to your specific task distribution. Frozen encoders (vision + language) provide general understanding.

---

## Challenge Exercises (Optional)

### Challenge 1: Morphology Transfer

You train a VLA policy on a 7-DOF robot. Deploy on 6-DOF robot without retraining.

**Approach**: Output task-space actions (x, y, z, gripper) instead of joint-space. Let IK solver handle morphology differences.

**Difficulty**: Medium (requires understanding IK)

### Challenge 2: Failure Mode Diagnosis

Collect 50 real robot trials and diagnose failures.

**Approach**:
1. Run vla_evaluation.py on your trials
2. Look at failure_analysis() output
3. Identify primary failure mode (perception, language, grounding, motor)
4. Propose specific fix

**Difficulty**: Medium (requires debugging skills)

### Challenge 3: Real-Time Inference

Policy needs <100ms latency but currently takes 2 seconds.

**Approach**: Model quantization or distillation to smaller model
- Quantization: Convert to 8-bit integers
- Distillation: Train small model to match large model
- Caching: Reuse predictions for similar scenes

**Difficulty**: Hard (requires optimization expertise)

---

## Summary

You now understand:

✅ **Zero-shot VLA**: Language models can reason about robot tasks from internet knowledge
✅ **Few-shot VLA**: Examples teach task format and parameter ranges
✅ **Fine-tuned VLA**: Specific data adapts general models to your robot
✅ **Evaluation**: Measure success, diagnose failures, compare approaches
✅ **Transfer**: Adapt to new robots, tasks, and embodiments

---

## Further Reading

- **RT-2 Paper**: https://robotics-transformer-2.github.io/
- **VoxPoser**: https://arxiv.org/abs/2307.05973
- **Code-as-Policies**: https://code-as-policies.github.io/
- **LLaVA**: https://github.com/haotian-liu/LLaVA

---

**Next Module**: [Module 5: End-to-End Learning & Diffusion Models](../05-embodied-learning/intro.md) (Coming soon)
