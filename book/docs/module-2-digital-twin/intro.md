---
sidebar_position: 1
title: "Module 2: The Digital Twin"
---

# Module 2: The Digital Twin (Gazebo & Unity)

## Overview

A **digital twin** is a virtual replica of a physical robot that behaves identically to its real-world counterpart. Before deploying AI policies to expensive hardware, we train, test, and validate in simulation. This module covers two complementary simulation platforms:

- **Gazebo**: Physics-accurate simulation for control and dynamics
- **Unity**: High-fidelity rendering for perception and human-robot interaction

## Learning Objectives

By the end of this module, you will be able to:

- Set up **Gazebo simulation** environments for humanoid robots
- Understand **physics engines** and their parameters
- Create robot models using **URDF** and **SDF** formats
- Simulate sensors: **LiDAR**, **depth cameras**, and **IMUs**
- Use **Unity** for photorealistic rendering and human interaction scenarios
- Understand the **sim-to-real gap** and mitigation strategies

## Why Both Gazebo and Unity?

| Aspect | Gazebo | Unity |
|--------|--------|-------|
| **Primary Use** | Physics & Control | Perception & Rendering |
| **Physics Accuracy** | High (ODE, Bullet, DART) | Medium (PhysX) |
| **Visual Fidelity** | Medium | High (HDRP) |
| **Sensor Simulation** | LiDAR, IMU, cameras | RGB cameras, depth |
| **Human Avatars** | Limited | Excellent |
| **VR/AR Support** | No | Yes |
| **ROS Integration** | Native (gazebo_ros) | Via ROS-TCP-Connector |

## The Simulation Development Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                  Simulation Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐     ┌──────────┐     ┌──────────────┐        │
│   │   URDF   │────▶│  Gazebo  │────▶│   Policy     │        │
│   │  Model   │     │  Physics │     │   Training   │        │
│   └──────────┘     └──────────┘     └──────┬───────┘        │
│                                            │                 │
│                                            ▼                 │
│   ┌──────────┐     ┌──────────┐     ┌──────────────┐        │
│   │  Unity   │◀────│  Trained │◀────│   Domain     │        │
│   │ Visuals  │     │  Policy  │     │   Random.    │        │
│   └──────────┘     └──────────┘     └──────────────┘        │
│                                            │                 │
│                                            ▼                 │
│                                     ┌──────────────┐        │
│                                     │  Real Robot  │        │
│                                     └──────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

### Week 6: Gazebo Simulation Fundamentals
- Gazebo architecture and physics engines
- World files and SDF format
- Physics parameters and timesteps

### Week 7: Sensor Simulation & Unity Integration
- Simulating LiDAR, cameras, and IMUs in Gazebo
- Unity setup with ROS-TCP-Connector
- High-fidelity human-robot interaction scenarios
- Domain randomization for robust policies

## The Sim-to-Real Gap

Training in simulation is powerful but introduces the **sim-to-real gap**—differences between simulated and real-world behavior:

| Gap Type | Simulation | Reality |
|----------|------------|---------|
| **Physics** | Perfect friction, rigid bodies | Deformable, variable friction |
| **Sensors** | Ideal measurements | Noise, calibration errors |
| **Actuators** | Instant response | Delays, backlash, wear |
| **Environment** | Static, controlled | Dynamic, unpredictable |

### Mitigation Strategies

1. **Domain Randomization**: Vary simulation parameters
2. **System Identification**: Measure real robot dynamics
3. **Sim-to-Real Fine-tuning**: Adapt policies on real hardware
4. **Photorealistic Rendering**: Reduce visual domain gap

## Prerequisites

- Completed Module 1 (ROS 2 fundamentals)
- Basic 3D graphics concepts
- Linux command line proficiency

## Software Requirements

- **Gazebo Fortress** (or Gazebo Harmonic)
- **Unity 2022.3 LTS** with HDRP
- **ROS-TCP-Connector** package for Unity
- **gazebo_ros_pkgs** for ROS 2 integration

## What's Next

This module transforms your ROS 2 knowledge into working simulations. You'll create environments where humanoid robots can learn to walk, balance, and interact—all without risking expensive hardware. The skills here directly feed into Module 3's NVIDIA Isaac platform for advanced AI training.

---

*Ready to build your robot's virtual world? Let's start with Gazebo fundamentals.*
