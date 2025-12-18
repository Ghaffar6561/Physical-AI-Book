# Glossary

This is a comprehensive glossary of terms used throughout the Physical AI & Humanoid Robotics textbook.

## A

**Action Grounding** — Mapping language concepts to motor commands or robot actions

**Affordance** — Physical property that suggests an action (e.g., handle on cup suggests "graspable")

**Autonomous System** — A system that operates independently without constant human intervention

## B

**Bridge Data** — Dataset containing robot demonstrations for learning manipulation skills

**Bullet** — Physics engine option in Gazebo, known for more accurate collision handling

## C

**Capstone Project** — The final integrative project that combines all modules of the textbook

**CLIP** — Contrastive Language-Image Pre-training model for connecting vision and language

**Code-as-Policies** — Approach where LLMs generate code to control robots

**Coordinate Frame** — Reference system for representing positions and orientations

## D

**Dense Neural Network** — Fully connected neural network without convolutional layers

**Domain Randomization** — Training with randomized parameters to build robustness to variation

**Domain Shift** — Change in data distribution (sim → real)

## E

**Embodied AI** — AI system with physical sensors/actuators (robot)

**Embodiment** — The physical form and sensorimotor capabilities of an agent

**End-to-End Learning** — Training complete system without modular components

## F

**Fine-Tuning** — Adapting a pre-trained model using a smaller dataset for a specific task

**Foundation Model** — Large model (GPT, Llama) trained on diverse data, fine-tuned for downstream tasks

**Forward Kinematics** — Computing end-effector position from joint angles

## G

**Gazebo** — Open-source 3D robotics simulator using ODE/Bullet physics

**gazebo_ros** — ROS 2 package providing bridge between Gazebo and ROS 2 topics/services

**Gaussian Noise** — Random noise following a normal distribution

**Gripper** — Robot end effector designed to grasp and hold objects

## H

**Hardware-in-the-Loop (HiL)** — Testing with real hardware in the control loop

**Humanoid Robot** — Robot with human-like form factor (torso, two arms, two legs)

## I

**Inertial Measurement Unit (IMU)** — Sensor measuring linear acceleration and angular velocity

**Inertia Tensor** — Mathematical representation of how mass is distributed relative to rotation axes

**Inverse Kinematics (IK)** — Computing joint angles needed to reach a target position

**Isaac Sim** — NVIDIA's simulation platform for robotics and autonomous machines

## L

**Latency Budget** — Maximum acceptable delay in perception-control loop

**Large Language Model (LLM)** — Transformer-based model trained on large text corpora

**LLaVA** — Large Language and Vision Assistant model combining vision and language

**LoRA** — Low-Rank Adaptation, a parameter-efficient fine-tuning method

## M

**Manipulation** — Robot skill involving grasping, moving, and controlling objects

**Modular Architecture** — System design with separate reasoning and control components

**Morphology** — Physical structure of robot (number of joints, sensor placement, size)

**Multimodal** — Model that processes multiple modalities (vision + language + proprioception)

## N

**Neural Network** — Computational model inspired by biological neural networks

**Noise Model** — Mathematical description of random variations in sensor measurements

## O

**Occupancy Map** — Grid-based representation of space indicating occupied/free areas

**ODE** — Open Dynamics Engine, a physics simulation engine used in Gazebo

**OpenAI GPT** — Generative Pre-trained Transformer model developed by OpenAI

**Ollama** — Local LLM serving framework

**ORCA** — A VLA system combining language models with robot control

## P

**Perception Pipeline** — Chain of sensor → filtering → feature extraction → decision

**Physical AI** — AI systems that interact with the physical world through sensors and actuators

**Prompt Engineering** — Designing text inputs to language models to elicit desired outputs

**Proprioception** — Sensing of robot's own body configuration and movement

## R

**Real-Time Factor** — Ratio of simulation time to real-world time (e.g., real_time_factor=1.0)

**Rolling Shutter** — Camera readout where rows are captured at different times

**ROS 2** — Robot Operating System version 2, middleware for robotic applications

**RT-1/RT-2** — Google's Robotics Transformer models

**Rviz** — 3D visualization tool for ROS

## S

**SDF** — Simulation Description Format; more detailed than URDF, includes world properties

**Semantic Gap** — Difference between high-level concepts and low-level robot actions

**Sensor Fusion** — Combining data from multiple sensors to reduce uncertainty

**Sim-to-Real Gap** — Difference between simulation behavior and reality

**Simulation Adequacy** — Whether simulation is accurate enough for specific task

**SLAM** — Simultaneous Localization and Mapping

**Spatial Reasoning** — Understanding and reasoning about positions and relationships in space

**State Estimation** — Determining robot's state (position, velocity, etc.) from sensor data

**System Identification** — Measuring real robot properties and modeling them

## T

**Task Decomposition** — Breaking high-level commands into low-level actions

**Timestep** — Simulation update interval in Gazebo

**Trajectory Planning** — Computing path through space and time for robot motion

## U

**URDF** — Unified Robot Description Format, XML format describing robot structure, kinematics, and sensors

**Unified Robot Description** — Standard format for describing robot geometry, kinematics, and dynamics

## V

**VLA (Vision-Language-Action)** — System combining vision, language understanding, and action execution

**VLVM (Vision-Language-Action Model)** — Neural network processing visual and language input to produce actions

**Vision Transformer (ViT)** — Transformer architecture applied to visual data

**Vision-Language Model (VLM)** — Neural network that understands both images and text

## W

**World File** — SDF file describing entire simulation environment in Gazebo

**Workspace** — Volume in space that robot can reach with its end effector

## Z

**Zero-Shot Learning** — Performing a task without specific training on that task