# Complete Training Pipeline: From Data to Deployment

End-to-end workflow for training robot manipulation policies.

---

## Full Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      EMBODIED LEARNING PIPELINE                  │
└─────────────────────────────────────────────────────────────────┘

STAGE 1: DATA COLLECTION
┌──────────────────────────────────────────┐
│ Collect robot demonstrations             │
│ ├─ Teleop (remote control)              │
│ ├─ Kinesthetic teaching (push arm)      │
│ └─ Self-play (prior policy)             │
│                                         │
│ Output: (image, action, reward) tuples  │
│ Time: 1-2 weeks                         │
└──────────────────────────────────────────┘
                    ↓

STAGE 2: DATA PROCESSING
┌──────────────────────────────────────────┐
│ Clean and organize data                  │
│ ├─ Remove failed trajectories            │
│ ├─ Normalize action ranges               │
│ ├─ Resample images (224×224)            │
│ └─ Train/val/test split (70/15/15)      │
│                                         │
│ Output: Preprocessed dataset            │
│ Time: 2-4 hours                         │
└──────────────────────────────────────────┘
                    ↓

STAGE 3: METHOD SELECTION
┌──────────────────────────────────────────┐
│ Choose learning paradigm:                │
│ ├─ BC (< 500 demos, fast)               │
│ ├─ Diffusion (500-2000 demos)           │
│ └─ RL (for refinement/exploration)      │
│                                         │
│ Decision: Based on time/data/performance│
│ Time: 30 minutes                        │
└──────────────────────────────────────────┘
                    ↓

STAGE 4: NETWORK DESIGN
┌──────────────────────────────────────────┐
│ Design neural network architecture       │
│ ├─ Input: image (224×224) + proprioception
│ ├─ Backbone: CNN or ViT                 │
│ ├─ Output: action (joint angles, gripper)
│ └─ Hidden: 256-512 neurons              │
│                                         │
│ Options:                                │
│ ├─ Small (2M params): Fast, simple     │
│ ├─ Medium (50M params): Balanced       │
│ └─ Large (500M params): Best quality   │
│                                         │
│ Time: 1-2 hours                         │
└──────────────────────────────────────────┘
                    ↓

STAGE 5: TRAINING
┌──────────────────────────────────────────┐
│ Train policy network                     │
│ ├─ BC: Supervised learning (MSE loss)  │
│ ├─ Diffusion: Predict noise (50-100 steps)
│ └─ RL: PPO with critic                  │
│                                         │
│ Hyperparameters:                        │
│ ├─ Batch size: 32-128                   │
│ ├─ Learning rate: 1e-3 to 1e-5         │
│ ├─ Epochs: 50-200                      │
│ └─ Validation every epoch               │
│                                         │
│ Time: 1 hour (BC) - 2 weeks (RL)       │
└──────────────────────────────────────────┘
                    ↓

STAGE 6: EVALUATION
┌──────────────────────────────────────────┐
│ Test policy on held-out data             │
│ ├─ Success rate (primary metric)        │
│ ├─ Confidence of predictions             │
│ ├─ Per-task performance                  │
│ └─ Failure analysis (wrong predictions) │
│                                         │
│ Thresholds:                             │
│ ├─ Good: >80% success                   │
│ ├─ Acceptable: >70% success             │
│ └─ Needs improvement: <70%              │
│                                         │
│ Output: Performance report               │
│ Time: 1-2 hours                         │
└──────────────────────────────────────────┘
                    ↓

                DECISION POINT
                    ↓
         ┌─────────────────────┐
         │ Performance OK?     │
         │ (>80% success)      │
         └─────────────────────┘
              ↙         ↘
          YES            NO
           ↓              ↓
        STAGE 7        ITERATE
        Deployment  ├─ Collect more data
                    ├─ Try different architecture
                    ├─ Adjust hyperparameters
                    └─ Back to STAGE 4/5

STAGE 7: REAL ROBOT TESTING (If applicable)
┌──────────────────────────────────────────┐
│ Validate on actual hardware              │
│ ├─ Run 20 trials on real robot          │
│ ├─ Compare sim vs real performance      │
│ └─ Identify sim-to-real gaps            │
│                                         │
│ If gap > 15%:                           │
│ ├─ Option A: Fine-tune with real data  │
│ ├─ Option B: Increase domain randomization
│ └─ Option C: Hybrid BC+RL               │
│                                         │
│ Time: 2-4 hours                         │
└──────────────────────────────────────────┘
                    ↓

STAGE 8: DEPLOYMENT
┌──────────────────────────────────────────┐
│ Package and deploy to production         │
│ ├─ Model quantization (reduce size)     │
│ ├─ Latency optimization                  │
│ ├─ Safety checks and fallbacks          │
│ └─ Monitoring and logging               │
│                                         │
│ Deployment checklist:                   │
│ ├─ ✓ Latency < budget (50-100ms)       │
│ ├─ ✓ Success rate > 80%                │
│ ├─ ✓ Fallback policy implemented       │
│ ├─ ✓ Graceful failure handling         │
│ └─ ✓ Real-time metrics logging         │
│                                         │
│ Time: 1-2 days                          │
└──────────────────────────────────────────┘
```

---

## Detailed: Data Collection Stage

```
┌─────────────────────────────────────────┐
│         DATA COLLECTION METHODS          │
└─────────────────────────────────────────┘

METHOD 1: TELEOPERATION
  Expert controls robot via interface
  ├─ Speed: 2-5 minutes per demonstration
  ├─ Cost: $50-150/hour expert time
  ├─ Quality: High (expert optimized)
  ├─ Data: 1000 hours → 10K-20K episodes
  └─ Best for: Complex manipulation

METHOD 2: KINESTHETIC TEACHING
  Human physically moves robot arm
  ├─ Speed: 1-2 minutes per demonstration
  ├─ Cost: $50-100/hour expert time
  ├─ Quality: Very high (natural motion)
  ├─ Data: 1000 hours → 30K-50K episodes
  └─ Best for: Learning natural motion

METHOD 3: SELF-PLAY
  Use previous policies to generate demos
  ├─ Speed: 1-5 seconds per episode (fast!)
  ├─ Cost: GPU compute only ($0.10/episode)
  ├─ Quality: Medium (biased toward prior)
  ├─ Data: 1000 GPU hours → 500K+ episodes
  └─ Best for: Scaling up existing policy

DATA COLLECTION TIMELINE:
├─ Goal: 500 grasping demonstrations
├─ Method: Teleop (2 hours)
├─ Quality check: 5% failure rate OK
├─ Total time: 20-30 hours teleoperation
├─ Cost: $1000-1500 (expert time)
└─ Result: High-quality dataset ready for training
```

---

## Detailed: Training Stage

### BC Training Loop

```
Training Behavioral Cloning
├─ Batch size: 32
├─ Learning rate: 1e-3 (with cosine annealing)
├─ Optimizer: Adam
├─ Loss: MSE + L2 regularization
│
├─ Epoch 1:
│  ├─ Train loss: 0.45
│  ├─ Val loss: 0.48
│  └─ Action error: 0.15 rad
│
├─ Epoch 20:
│  ├─ Train loss: 0.12
│  ├─ Val loss: 0.18
│  └─ Action error: 0.04 rad
│
├─ Epoch 50 (STOP - early stopping):
│  ├─ Train loss: 0.08
│  ├─ Val loss: 0.16 (no improvement)
│  ├─ 10 epochs without improvement → STOP
│  └─ Load best model from epoch 40
│
└─ Result: Network achieves 0.04 rad error on validation
```

### Diffusion Training Loop

```
Training Diffusion Policy
├─ Batch size: 64
├─ Learning rate: 1e-4
├─ Diffusion steps: 1000 (forward), 50-100 (inference)
├─ Noise schedule: Linear cosine
│
├─ Epoch 1:
│  ├─ Noise prediction loss: 0.85
│  └─ Validation: 0.88
│
├─ Epoch 50:
│  ├─ Noise prediction loss: 0.15
│  └─ Validation: 0.18
│  └─ Inference success rate: 72%
│
├─ Epoch 100:
│  ├─ Noise prediction loss: 0.08
│  └─ Validation: 0.12
│  └─ Inference success rate: 87%
│
└─ Result: Diffusion policy achieves 87% success
```

### RL Training Loop

```
Training RL Policy (PPO)
├─ Workers: 4 parallel environments
├─ Episodes per update: 2000 transitions
├─ Epochs per update: 3 passes through data
├─ Learning rate: 1e-4 (fixed, no schedule)
│
├─ Week 1 (random exploration):
│  ├─ Average return: -50 (very bad)
│  ├─ Success rate: 5%
│  └─ Agent exploring randomly
│
├─ Week 2 (initial learning):
│  ├─ Average return: -5 (better)
│  ├─ Success rate: 40%
│  └─ Policy starting to work
│
├─ Week 3 (convergence):
│  ├─ Average return: 0.5 (good)
│  ├─ Success rate: 88%
│  └─ Training plateau, convergence
│
├─ Evaluation on test set:
│  ├─ In-distribution: 94% success
│  ├─ Novel objects: 82% success
│  └─ Different lighting: 79% success
│
└─ Result: RL policy exceeds BC baseline (75%)
```

---

## Decision Tree: Which Approach?

```
START: Want to train a policy?
  │
  ├─ Have demonstrations? (100+)
  │  │
  │  ├─ YES, simple task (grasping)
  │  │  └─ CHOICE: Behavioral Cloning
  │  │     Time: 8 hours total
  │  │     Result: 70-80% success
  │  │
  │  ├─ YES, complex task (multimodal)
  │  │  └─ CHOICE: Diffusion Policy
  │  │     Time: 2-3 days total
  │  │     Result: 85-90% success
  │  │
  │  └─ YES, want best possible
  │     └─ CHOICE: BC + RL fine-tuning
  │        Time: 2-3 weeks total
  │        Result: 92-95% success
  │
  └─ NO demonstrations
     │
     ├─ Have simulation?
     │  │
     │  ├─ YES, simple task
     │  │  └─ CHOICE: PPO in simulation
     │  │     Time: 1-2 weeks
     │  │     Result: 80% in sim, 50% on real (sim-to-real gap)
     │  │
     │  └─ YES, use domain randomization
     │     └─ CHOICE: PPO + domain randomization
     │        Time: 2-4 weeks
     │        Result: 70% in sim, 65% on real
     │
     └─ NO simulation
        └─ CHOICE: Collect demonstrations first!
           "You can't train RL on real robot without good prior"
```

---

## Evaluation Metrics

```
┌──────────────────────────────────────────────┐
│     HOW TO EVALUATE YOUR POLICY              │
└──────────────────────────────────────────────┘

PRIMARY METRIC: Success Rate
├─ Definition: % of test trials that succeeded
├─ Target: >80% for deployment
├─ Example: 48/50 trials succeeded = 96% success
└─ Interpretation: Can you run 100 tasks, expect ~96 to work

SECONDARY METRICS:

1. Confidence
   ├─ Network prediction uncertainty
   ├─ High confidence + failure = bad (overconfident)
   ├─ Low confidence + success = wasted potential
   └─ Target: Confidence ~= success rate

2. Generalization
   ├─ Test on novel objects (not in training)
   ├─ Test on distribution shift (different lighting/angles)
   ├─ Test on different robots (different morphology)
   └─ Target: >70% success on novel objects

3. Speed
   ├─ Inference latency (how fast predictions)
   ├─ BC: 50-100ms
   ├─ Diffusion: 200-500ms
   ├─ RL: 100-150ms
   └─ Target: <200ms for real-time tasks

4. Robustness
   ├─ Variance across trials (high variance = unreliable)
   ├─ Measure: std dev of success rate across 10 runs
   ├─ Example: 85% ± 3% (good), 85% ± 15% (bad)
   └─ Target: Variance < 5%

5. Failure Analysis
   ├─ Categorize failures:
   │  ├─ Perception (misread object position)
   │  ├─ Language understanding (wrong instruction interpretation)
   │  ├─ Motor control (physically impossible move)
   │  └─ Environment (unforeseen obstacle)
   │
   └─ Target: <5% perception errors (most important to fix)
```

---

## Common Pitfalls & Solutions

```
┌──────────────────────────────────────────────┐
│        DEBUGGING YOUR TRAINING                │
└──────────────────────────────────────────────┘

PITFALL 1: Training loss decreases, but test performance is poor
└─ Cause: Distribution mismatch (BC) or overfitting
└─ Solution:
   ├─ Check if network can memorize training data
   ├─ If yes: Data quality issue (bad labels)
   ├─ If no: Network too small
   └─ Add: Data augmentation, regularization, more data

PITFALL 2: Training is too slow
└─ Cause: Network too large or slow hardware
└─ Solution:
   ├─ Profile: Which stage is bottleneck?
   ├─ Data loading: Use num_workers=4 in DataLoader
   ├─ Forward pass: Use smaller network temporarily
   └─ GPU memory: Reduce batch size

PITFALL 3: Success rate plateaus at 60-70% (BC)
└─ Cause: Fundamental limit of behavioral cloning
└─ Solution:
   ├─ Try DAgger (expert corrections)
   ├─ Switch to Diffusion Policy
   ├─ Or fine-tune with RL
   └─ Accept limitation: BC not suitable for this task

PITFALL 4: Policy works in simulation, fails on real robot
└─ Cause: Sim-to-real gap
└─ Solution:
   ├─ Increase domain randomization in simulation
   ├─ Fine-tune on small set of real data (50 trials)
   ├─ Use RL for adaptation
   └─ Or: Accept 15-20% drop in performance

PITFALL 5: RL training diverges (success drops to 0%)
└─ Cause: Learning rate too high or reward is bad
└─ Solution:
   ├─ Check reward computation (is it sensible?)
   ├─ Reduce learning rate by 10×
   ├─ Verify environment is not broken
   └─ Reload good checkpoint and continue
```

---

## Timeline Example: Grasping Task

```
TIMELINE: Complete grasping from 0 to deployment

Week 1-2: Setup & Data Collection
│ Day 1-2:   Collect 500 grasping demonstrations (teleop)
│ Day 3-4:   Inspect data, remove bad trials
│ Day 5-7:   Preprocessing and train/val/test split
│ Result:    Clean dataset ready for training

Week 3: Behavioral Cloning
│ Day 1:     Design network (CNN baseline)
│ Day 2:     Implement data loading and training loop
│ Day 3:     Train for 50 epochs (1 hour), evaluate (2 hours)
│ Day 4:     Tune hyperparameters (learning rate, etc)
│ Day 5:     Test on real robot (20 trials)
│ Result:    73% success (acceptable but not great)

Week 4: Improve with Diffusion
│ Day 1-2:   Implement diffusion policy
│ Day 3-4:   Train (8 hours), find good step count
│ Day 5:     Test on real robot (20 trials)
│ Result:    85% success (much better!)

Week 5: Deploy
│ Day 1:     Optimization and packaging
│ Day 2-3:   Real robot validation (50 trials)
│ Day 4-5:   Set up monitoring, fallbacks
│ Result:    Production ready at 86% success

TOTAL TIME: 5 weeks from zero to deployment
EFFORT: 1 person full-time
COST: ~$2000 (expert time for data collection)
RESULT: Reliable grasping at 86% success
```

---

## Key Takeaways

✅ **Stage 1-2: Collection & Preprocessing** → Determines data quality
✅ **Stage 3: Method Selection** → Determines timeline and performance ceiling
✅ **Stage 4-5: Design & Training** → Requires iteration and debugging
✅ **Stage 6: Evaluation** → Use proper metrics, don't overfit to training data
✅ **Stage 7-8: Testing & Deployment** → Sim-to-real gap is real, plan for it

---

## Quick Decision Guide

```
How much time do you have?
├─ 1 day: BC only, 70% success
├─ 1 week: BC + light RL tuning, 80% success
├─ 2 weeks: Diffusion Policy, 87% success
└─ 1 month: BC + full RL fine-tuning, 92% success
```

---

**Next:** [Back to Learning Overview →](intro.md)

