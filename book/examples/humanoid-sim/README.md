# Autonomous Humanoid Robot Capstone Project

This directory contains the complete capstone project: an autonomous humanoid robot that executes spoken commands in Gazebo simulation.

## Structure

- `ros2_nodes/` — ROS 2 node implementations (perception, planning, control, voice)
- `gazebo_models/` — URDF and Gazebo models of the humanoid and environment
- `perception/` — Computer vision and SLAM algorithms
- `planning/` — Task planning and motion planning
- `vla/` — Voice interface and LLM integration

## Quick Start

```bash
cd book/examples/humanoid-sim
python -m pip install -r ../requirements.txt
python launch_humanoid.py
```

See the main book for detailed setup and usage instructions.
