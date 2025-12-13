# Gazebo Fundamentals

Gazebo is an open-source 3D robotics simulator. It provides:

- **Physics simulation** — Realistic dynamics, collision detection
- **Sensor simulation** — Cameras, LiDAR, IMU with configurable noise
- **ROS 2 integration** — Native communication with ROS 2 nodes
- **Visualization** — 3D GUI for viewing simulations
- **Headless mode** — Run without GUI for CI/testing

## Basic Concepts

### World Files (SDF)

A world file (.world or .sdf) describes:
- Ground plane and static objects
- Physics engine settings (gravity, friction)
- Lighting and visual appearance
- Plugins (ROS 2 bridges, sensors)

### Models

A model is a collection of links and joints. Models can be:
- **Static** — Ground, walls, furniture
- **Dynamic** — Robots, balls, objects

### Plugins

Plugins connect Gazebo to external code (ROS 2). They:
- Publish sensor data to ROS 2 topics
- Subscribe to control commands from ROS 2
- Implement custom physics or sensor behavior

## Your First Simulation

To come in exercises.

---

**Next**: [URDF & Humanoids](urdf-humanoid.md)
