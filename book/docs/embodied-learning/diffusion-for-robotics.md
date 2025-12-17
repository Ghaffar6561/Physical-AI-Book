# Diffusion Models for Robot Control

**Key insight**: The same technology that powers DALL-E (image generation) can generate robot trajectories. Instead of generating pixels, we generate smooth, collision-free action sequences.

---

## Background: How Diffusion Works

Diffusion models generate data in two phases:

### Phase 1: Forward Process (Corruption)
```
Real data (trajectory):  τ = [a₀, a₁, ..., a₁₀₀]
                              ↓ add noise
Step 1:                   τ₁ = τ + ε₁
                              ↓ add more noise
Step 2:                   τ₂ = τ₁ + ε₂
                              ↓ add more noise
...
Step 1000:                τ₁₀₀₀ = pure Gaussian noise

Trajectory is destroyed over 1000 steps.
```

### Phase 2: Reverse Process (Generation)
```
Pure noise:               τ₁₀₀₀ ~ N(0, I)
                              ↓ remove noise (network predicts ε)
Step 999:                 τ₉₉₉ = τ₁₀₀₀ - predicted_noise
                              ↓ remove more noise
Step 998:                 τ₉₉₈ = τ₉₉₉ - predicted_noise
                              ↓ remove more noise
...
Step 0:                   τ₀ = valid trajectory

Noise is removed over 1000 steps → trajectory is refined.
```

### Mathematical Formulation

**Forward process** (t: 0 → T = 1000):
```
x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε_t

where:
- ᾱ_t = product of noise schedules (controls corruption speed)
- x_0 = original trajectory
- ε_t ~ N(0, I) = random noise
```

**Reverse process** (t: T → 0):
```
x_{t-1} = (1/√α_t) · (x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t · z

where:
- ε_θ = neural network that predicts noise
- This is the KEY: network learns to denoise
```

---

## Application to Robotics: Diffusion Policy

### Core Idea

```
Input: observation (image + proprioception)
Task: Generate smooth trajectory [a₀, a₁, ..., a₁₀]

Step 1: Initialize with random noise τ ~ N(0, I)
Step 2: Condition network on observation
        τ̂ = diffusion_network(τ, observation, t=1000)
Step 3: Denoise iteratively
        for t in [1000, 999, ..., 1]:
            ε_predicted = network(τ, observation, timestep=t)
            τ = τ - ε_predicted
Step 4: Output final trajectory τ₀
        Execute as: a₀, a₁, ..., a₁₀

Result: Smooth, physically plausible trajectory generated from observation.
```

### Why This Works for Robotics

**Advantage 1: Handles Multimodality**

Real tasks have multiple valid solutions:
```
Pick up cup:
- Approach from north (45% of successful grasps)
- Approach from east (40%)
- Approach from south (15%)

Behavioral cloning picks average direction → fails most of the time
Diffusion model generates one valid direction per inference → success
```

**Advantage 2: Generates Smooth Trajectories**

```
BC: [0.3, 0.5, 0.2, 0.1, -0.8, ...]  ← Jerky, discontinuous
                                         Can cause robot to vibrate

Diffusion: [0.3, 0.32, 0.35, 0.38, ...]  ← Smooth, realistic
                                             Natural robot motion
```

**Advantage 3: Implicit Constraints**

Network implicitly learns:
- Joint limits (don't predict θ > 180°)
- Collision avoidance (trajectory avoids table)
- Momentum (smooth velocity changes)

Without explicit constraint checking.

---

## Architecture: Diffusion Policy

### Network Structure

```
Input:
  - observation: RGB image (224×224) + proprioception (7D joint angles)
  - timestep: t ∈ {0, 1, ..., 1000}
  - current_noise: τ_t

Vision encoder:
  CNN or ViT → 256D features

Proprioception encoder:
  FC layer: 7D → 64D

Timestep embedding:
  Sinusoidal embedding → 128D

Fusion:
  Concatenate [vision_features, prop_features, timestep_embedding]
  → MLP(256+64+128 → 512 → 512 → 256)

Output:
  Predicted noise ε̂_t same shape as trajectory (10 actions × 4D = 40D)
```

### Training

```python
# Training loop
for batch in dataloader:
    observations, trajectories = batch
    # trajectories shape: (batch_size, sequence_length=10, action_dim=4)

    # Sample random timestep
    t = random(0, T)

    # Add noise to trajectory
    τ_t = √(ᾱ_t) * trajectories + √(1 - ᾱ_t) * noise

    # Predict noise
    ε_pred = network(τ_t, observations, t)

    # Loss: how well did we predict the noise?
    loss = MSE(ε_pred, noise)

    loss.backward()
    optimizer.step()
```

**Key insight**: Network learns to reverse the diffusion process by predicting noise.

---

## Inference: Generating Trajectories

```python
def infer(observation):
    # Start with random noise
    τ = normal(0, 1, shape=(trajectory_length=10, action_dim=4))

    # Iteratively denoise
    for t in [1000, 999, ..., 1]:
        # Predict noise at this timestep
        ε_pred = network(τ, observation, timestep=t)

        # Remove predicted noise from trajectory
        # Using DDPM equation
        α = alpha_schedule[t]
        ᾱ = cum_alpha_schedule[t]

        τ = (1/√α) * (τ - (1-α)/√(1-ᾱ) * ε_pred)

        # Add small noise for next step (except last step)
        if t > 1:
            τ += √(variance_schedule[t]) * normal(0, 1)

    return τ  # Final trajectory [a₀, a₁, ..., a₉]
```

---

## Real-World Results: Diffusion Policy

### Benchmark Study

**Researchers**: UC Berkeley (Chi et al., 2023)

**Setup**:
- 6 manipulation tasks: grasping, pushing, insertion, etc.
- 50-300 demonstrations per task
- Compare: BC, Diffusion Policy, RL

| Task | BC Success | Diffusion | RL | Notes |
|------|-----------|-----------|----|----|
| **Grasp Seen Object** | 50% | 85% | 88% | Diffusion handles multimodality |
| **Grasp Novel Object** | 30% | 70% | 75% | Diffusion generalizes better |
| **Push Block** | 45% | 80% | 82% | Smooth trajectory helps |
| **Insert Peg** | 35% | 75% | 80% | Requires fine control |
| **Open Drawer** | 40% | 78% | 85% | Long horizon helps RL |

**Key finding**: Diffusion Policy achieved 80-85% average success with only 100-300 demonstrations. BC only got 40-50%.

### Why the Gap?

1. **Multimodal grasping**: Cup can be grasped from multiple angles
   - BC picks average → fails
   - Diffusion generates one valid angle → succeeds

2. **Smooth trajectories**: Diffusion generates natural motion
   - BC jumps between actions → jerky
   - Diffusion interpolates → smooth

3. **Out-of-distribution robustness**: Diffusion trained on noise, more robust
   - BC trains only on clean demonstrations
   - Diffusion trains on progressively corrupted trajectories

---

## Comparison: BC vs Diffusion vs RL

| Metric | BC | Diffusion | RL |
|--------|----|-----------|----|
| **Data needed** | 100 demos | 200 demos | 10K trials |
| **Training time** | 1 hour | 4-8 hours | 1-4 weeks |
| **Inference time** | 50ms | 200-500ms | 100ms |
| **Success rate** | 50-70% | 80-90% | 90-95% |
| **Generalization** | Poor (30-50% novel) | Good (60-75% novel) | Best (75-85% novel) |
| **Multimodal actions** | Fails | Handles well | Handles well |
| **Smoothness** | Jerky | Smooth | Variable |
| **Debugging** | Easy | Medium | Hard |

---

## Speed vs Quality Tradeoff

Diffusion quality depends on number of denoising steps:

```
Steps | Speed | Quality | Use Case
------|-------|---------|----------
10    | 50ms  | Fair    | Real-time, sim only
50    | 150ms | Good    | Real robot, moderate precision
100   | 250ms | Very good | Complex manipulation
1000  | 5s    | Excellent | Offline analysis
```

**Recommendation for real robots**: 50-100 steps (150-300ms latency)

**For comparison**:
- BC inference: 50ms (3-6× faster)
- RL inference: 100ms (similar speed)

**Why diffusion is slower**:
- BC: Single forward pass
- Diffusion: 50-100 sequential forward passes
- But each pass is cheaper than RL network

---

## Advanced: Conditioning Strategies

### Observation Conditioning

**Current approach**: Concatenate observation features
```
τ = network(τ_t, observation_features)
```

**Better approach**: Cross-attention
```
observation_features → query
τ_t → key/value
Cross-attention: condition diffusion on which parts of image matter
```

Result: 5-10% improvement on novel objects

### Goal Conditioning

**Learn trajectories to reach goal**:
```
network(τ_t, current_observation, goal_image)
```

Example:
```
Current state: Cup is at position A
Goal state: Cup is at position B
Network generates trajectory to move cup from A → B
```

### Text Conditioning

**Combine language with diffusion**:
```
network(τ_t, image, language_embedding)
```

Example:
```
Instruction: "Grasp gently" vs "Grasp firmly"
Diffusion generates trajectories with different gripper forces
```

---

## Practical Deployment

### Latency Budget

Real-time robot control needs < 100ms per action

```
Diffusion Policy Timing:
- Image preprocessing: 5ms
- Feature extraction: 10ms
- Denoising steps (50 steps): 150-200ms  ← Problem!
- Action execution: 10ms
Total: 175-225ms ← Too slow for real-time control

Solution 1: Reduce steps to 10-20 (faster, lower quality)
Solution 2: Use smaller model (same steps, faster inference)
Solution 3: Use RL instead (single forward pass, 100ms)
```

### Deployment Checklist

- [ ] Measure latency on target hardware (GPU available?)
- [ ] Test with network failures (what if network drops frame?)
- [ ] Batch inference (accumulate 5 frames, denoise once)
- [ ] Fallback policy (what if diffusion fails? Have BC backup)
- [ ] Monitor success rate (unexpected failures → retrain)

---

## Debugging Diffusion Policies

### Problem 1: Training is Slow

```
Symptom: Loss plateaus after 100 epochs
Root cause: Timestep sampling is too uniform
Solution: Sample more important timesteps (middle of noise schedule)
```

### Problem 2: Generates Strange Trajectories

```
Symptom: Valid action sequences, but jerky or unlikely
Root cause: Too few denoising steps during inference
Solution: Increase steps from 50 → 100
Trade-off: Slower inference (200ms → 300ms)
```

### Problem 3: Overfits Training Data

```
Symptom: 95% success on training tasks, 40% on novel objects
Root cause: Model too large or network capacity issue
Solution: Regularize network, add more diverse demonstrations
```

### Problem 4: Inference is Too Slow

```
Symptom: 300ms latency, need < 100ms
Root cause: Too many denoising steps
Solution: Reduce to 10 steps (less quality but 10× faster)
Alternative: Use distillation (train small BC from diffusion)
```

---

## When to Use Diffusion

### Use Diffusion When:

✅ Task has multiple valid solutions (many grasping angles)
✅ Need smooth, natural trajectories
✅ Have 100-500 demonstrations available
✅ Can tolerate 150-500ms latency
✅ Want good out-of-distribution generalization
✅ Don't need real-time control

### Use BC Instead When:

✅ Need < 50ms latency
✅ Task has single clear solution
✅ Limited demonstrations (< 100)
✅ Inference hardware is weak (embedded systems)
✅ Can afford DAgger interventions

### Use RL Instead When:

✅ Need best possible performance (90-95%)
✅ Have time for 1-4 weeks training
✅ Can generate ~10K synthetic trials
✅ Task is complex, requires exploration
✅ No good demonstrations available

---

## Key Takeaways

✅ **Diffusion reverses noise** - trains by predicting noise, inverts during inference
✅ **Multimodal trajectories** - naturally handles multiple valid solutions
✅ **Smooth outputs** - diffusion penalizes jerky motion
✅ **Better generalization** - 80-90% success vs BC's 50-70%
✅ **Trade latency for quality** - fewer steps = faster but lower quality
✅ **Slower than BC** - 50-100 forward passes vs 1

---

## Next Steps

1. **Understand the noise schedule** - Control corruption speed during training
2. **Experiment with step counts** - Find your latency/quality sweet spot
3. **Compare to behavioral cloning** - See the multimodality advantage
4. **Try on your task** - Collect demonstrations, train diffusion policy

---

## Further Reading

- **Denoising Diffusion Probabilistic Models** (Ho et al., 2020): Original DDPM paper
- **Diffusion Policy** (Chi et al., 2023): Application to manipulation, comprehensive benchmarks
- **Diffusion Models as Generative Priors** (Sohl-Dickstein et al., 2015): Theoretical foundations
- **Score-Based Generative Modeling** (Song et al., 2020): Alternative formulation

---

**Next Section:** [Reinforcement Learning for Embodied AI →](reinforcement-learning.md)

