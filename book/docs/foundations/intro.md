# Module 1: Physical AI Foundations

## Overview

This module teaches the **conceptual foundations** of Physical AI: why robots are fundamentally different from digital AI systems, and what makes simulation essential for embodied intelligence.

**Target audience**: Senior CS students with machine learning background but minimal robotics experience.

**Time commitment**: 2-3 hours reading + 1-2 hours exercises.

## Learning Objectives

By the end of this module, you will understand:

### Core Concepts
- [ ] **Embodied Intelligence** — How sensors, actuators, and physics shape intelligence
- [ ] **Perception-Action Loops** — The closed-loop feedback between robot and environment
- [ ] **Real-Time Constraints** — Why milliseconds matter in robotics
- [ ] **Simulation-Reality Gap** — What simulations can/cannot capture

### Practical Knowledge
- [ ] **ROS 2 Basics** — Nodes, topics, services, and actions
- [ ] **Robot Architectures** — How perception, planning, and control integrate
- [ ] **Why Simulation Works** — Digital twins as development accelerators
- [ ] **Failure Modes** — Common mistakes when transitioning from digital AI to robotics

### Independent Test (US1)
You can **independently** demonstrate:
- ✅ Explain 3+ differences between digital AI and physical AI
- ✅ Describe a sensor-action loop for a given task
- ✅ Argue why simulation is essential for safety and iteration
- ✅ Design a simple ROS 2 publisher-subscriber system

## Module Structure

### Part 1: Why Physical AI is Different
**Embodied Intelligence** — The foundational concepts

1. **Embodied Intelligence** (conceptual)
   - Definition and philosophical grounding
   - Comparison: Digital AI vs Physical AI
   - Examples: Chess engine vs robot grasping

2. **Perception-Action Loops** (conceptual + visual)
   - Feedback mechanisms
   - Sensors, computation, control, actuation
   - Why closed loops are essential

### Part 2: Building Robot Systems
**ROS 2 Foundations** — Software for embodied intelligence

3. **ROS 2 Introduction** (conceptual + code)
   - Nodes, topics, services, actions
   - Why distributed computation matters
   - First examples: pub/sub, services

4. **Exercises** (hands-on)
   - Design perception-action loops
   - Write ROS 2 nodes
   - Understand system architecture

## Why This Module Matters

### For AI/ML Practitioners
If you're coming from machine learning, this module **rewires your intuition**. Your experience with:
- **Batch processing** → Must change to **real-time pipelines**
- **Training on datasets** → Must shift to **deploying on hardware**
- **Offline optimization** → Must adapt to **online closed-loop control**

Without understanding these differences, you'll design systems that work in simulation but fail on real robots.

### For Software Engineers
If you're coming from software engineering, robotics adds:
- **Physical constraints** — Timing, power, mechanical wear
- **Safety concerns** — Hardware can hurt people
- **Hardware variation** — No two robots behave identically
- **Real-time requirements** — Guarantees matter more than throughput

### For Roboticists
If you're familiar with robots, this module provides **modern AI context**. We'll show:
- How large language models interface with robot control
- Why simulation fidelity matters for neural networks
- How to bridge the sim-to-real gap
- Integration patterns for perception, planning, and action

## Key Insight: The Embodied Intelligence Perspective

**Embodied intelligence** is the understanding that:

> Intelligence arises from the **interaction** between an organism and its **environment** through **sensors** and **actuators**, not from abstract computation alone.

This is NOT "embodied" = "has a body." Instead, it means:
- Intelligence is **grounded in physical reality**
- Perception and action **co-evolve**
- **Time and dynamics** are fundamental constraints
- **Safety** is not optional—it's architectural

## Learning Path Recommendation

### Recommended Order (Sequential)
1. Read **Embodied Intelligence** (30 min)
2. Skim **ROS 2 Introduction** (20 min)
3. Try exercises (30 min)
4. Code examples: simple pub/sub (30 min)

### Time-Boxed Path (90 minutes)
- Read Embodied Intelligence: 20 min
- Read ROS 2 core concepts only: 15 min
- Do exercises 1-2: 30 min
- Run one code example: 25 min

### Deep Dive Path (2-3 hours)
- Read all sections
- Do all exercises with write-ups
- Run all code examples and modify them
- Design your own ROS 2 system (30 min)

## What You'll Build

By the end of this module, you'll understand the architecture of a humanoid robot system:

```
PERCEPTION          PLANNING           CONTROL
Sensors       →  State Estimation →  Task Planning →  Motor Commands
(camera,                              (where to go,    (joint angles,
 lidar,         + Localization       what to do)       gripper force)
 imu)                                                    ↓
  ↓                   ↓                  ↓          Physical Robot
  └─────────────────────────────────────────────────────┘
         Closed-Loop Feedback & Adaptation
```

This feedback loop is **the essence of embodied intelligence**. Without it, your robot can't adapt to unexpected situations.

## Assumptions

**Required Background**:
- Solid Python programming (classes, modules, decorators)
- Familiarity with Linux command line (cd, ls, grep, cat)
- Understanding of neural networks (forward pass, loss, backprop)
- Basic understanding of probability and statistics

**Not Required** (but helpful):
- Prior robotics experience
- Knowledge of control theory
- Familiarity with Gazebo or ROS 1

**We'll explain from first principles**: No robotics jargon without introduction.

## Glossary (Quick Reference)

| Term | Definition |
|------|-----------|
| **Node** | A ROS 2 process that performs computation |
| **Topic** | A named channel for asynchronous pub/sub messages |
| **Service** | Synchronous request-response between nodes |
| **Action** | Long-running task with progress feedback |
| **Embodied Intelligence** | Intelligence arising from interaction with physical world |
| **Digital Twin** | Software simulation of a robot and environment |
| **Sim-to-Real** | Transferring policies from simulation to real hardware |
| **Perception-Action Loop** | Closed feedback from sensors → decision → action → new sensors |

## What's Next?

After completing this module, you'll have the **conceptual foundations** needed for:
- **Module 2**: Modeling and simulating robots (Gazebo)
- **Module 3**: Perception pipelines and sim-to-real transfer
- **Module 4**: Vision-Language-Action systems (language + robotics)
- **Capstone**: Building a complete autonomous humanoid

---

## How to Use This Module

**Reading**: Use the sidebar to navigate between sections. Click links to jump to related content.

**Code Examples**: All examples are executable Python. Run them with:
```bash
python examples/[example-name].py
```

**Exercises**: Each exercise has a solution. Try to solve it first, then check your answer.

**Diagrams**: SVG diagrams are interactive on the web. Hover to see labels.

---

**Ready to start?** → [Embodied Intelligence](embodied-intelligence.md)
