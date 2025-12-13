# Research Findings: Physical AI & Humanoid Robotics Book

**Date**: 2025-12-14
**Branch**: `001-physical-ai-book`
**Conducted For**: Implementation Planning (Phase 0)

## 1. Docusaurus Deployment & GitHub Pages

**Decision**: Use Docusaurus 3.x with GitHub Pages deployment

**Rationale**:
- Docusaurus is purpose-built for technical documentation and supports Markdown natively
- GitHub Pages integration is seamless and cost-free
- Supports versioning, search, dark mode, and offline capabilities out-of-the-box
- Large ecosystem of plugins for diagrams, code syntax highlighting, and MDX support

**Alternatives Considered**:
- **GitBook**: Simpler UI but less control over customization and build process
- **MkDocs**: Good for simpler docs; less powerful than Docusaurus for a complex book structure
- **Custom static site**: Would require significant infrastructure investment; Docusaurus provides this for free

**Best Practices**:
- Use `docusaurus.config.js` to define site metadata, sidebar structure, and plugins
- Enable code block syntax highlighting for Python, YAML, and shell commands
- Use `sidebars.js` for explicit module ordering to enforce learning progression
- Implement search indexing for reader discoverability
- Deploy via GitHub Actions workflow triggering on main branch pushes

## 2. ROS 2 Teaching & Learning Progression

**Decision**: Introduce ROS 2 concepts in layers: nodes → topics → services → actions, with hands-on Python examples

**Rationale**:
- Progressive layering matches how roboticists think about distributed systems
- Python (rclpy) is more accessible to learners than C++ (rclcpp) and suitable for this audience
- Practical examples with actual pub/sub patterns reinforce theoretical understanding

**Key Teaching Elements**:
1. **Nodes**: Processes that perform computation; use simple pub/sub example (sensor publisher → logger subscriber)
2. **Topics**: Asynchronous, many-to-many message passing; show publisher/subscriber lifecycle
3. **Services**: Synchronous request/response; teach when to use vs. topics
4. **Actions**: Long-running tasks with feedback; essential for robot control (move arm, navigate)
5. **Parameter Server**: Centralized configuration management for robot settings

**Alternatives Considered**:
- **C++ first**: More performant but steeper learning curve; Python is adequate for educational context
- **Omit ROS 2 fundamentals**: Spec requires FR-001; foundational knowledge is non-negotiable

**Best Practices**:
- Start with minimal working examples (5-10 lines of code)
- Use `rclpy.init()` / `rclpy.spin()` pattern consistently
- Show common debugging: `ros2 topic list`, `ros2 node info`, message inspection
- Provide docker/devcontainer setup to ensure consistent ROS 2 environment across student machines

## 3. Gazebo for Pedagogical Simulation

**Decision**: Use Gazebo 11+ as the primary simulation platform; cover NVIDIA Isaac as advanced topic

**Rationale**:
- Gazebo is free, open-source, and widely used in robotics education
- Physics engine (ODE, Bullet) suitable for teaching dynamics and sensor simulation
- URDF/SDF markup is standard across ROS 2 ecosystem
- Lower computational barrier than NVIDIA Isaac (runs on CPU with GPU acceleration optional)

**Key Teaching Elements**:
1. **URDF Models**: Teach link/joint hierarchy, collision geometry, inertia properties
2. **Physics Parameters**: Gravity, friction, damping—concepts that affect real robots
3. **Sensor Simulation**: Cameras, LiDAR, IMU with realistic noise and timing
4. **World Files**: Environment setup, object placement, lighting
5. **Gazebo Plugins**: Custom code for robot behavior and sensor processing

**Alternatives Considered**:
- **NVIDIA Isaac Sim first**: More photorealistic but higher barrier to entry; defer to Module 3 as advanced path
- **CoppeliaSim**: Good alternative but smaller community; Gazebo is industry standard

**Best Practices**:
- Provide pre-built URDF files for humanoid models (e.g., Boston Dynamics Atlas URDF, simplified custom model)
- Use `gazebo_ros` package to bridge ROS 2 and Gazebo communication
- Show headless simulation (non-GUI) for CI/testing
- Benchmark simulation speed on typical student hardware; target 60+ Hz for humanoid physics

## 4. NVIDIA Isaac Sim: Capabilities & Integration

**Decision**: Introduce NVIDIA Isaac as photorealistic simulation platform in Module 3; focus on synthetic data generation and domain randomization

**Rationale**:
- Isaac provides photorealistic rendering suitable for vision-based learning
- Supports domain randomization essential for sim-to-real transfer
- Scales to large-scale synthetic data generation (1000s of images/trajectories)
- Cloud-based option available for students without local GPUs

**Key Teaching Elements**:
1. **Photorealistic Rendering**: Why visual fidelity matters for vision models
2. **Synthetic Data Generation**: Creating labeled datasets for training perception models
3. **Domain Randomization**: Randomizing textures, lighting, object positions to improve transfer
4. **Extensions SDK**: Custom Python code integration for perception pipelines
5. **Isaac ROS Integration**: Connecting Isaac output to ROS 2 nodes

**Alternatives Considered**:
- **Omit photorealistic simulation**: Valid for basic humanoid control but limits advanced perception teaching
- **UnityRobotics**: Good alternative but smaller ROS integration ecosystem

**Best Practices**:
- Provide step-by-step tutorials for Isaac Sim installation and environment setup
- Offer reduced-fidelity alternatives for students without high-end GPUs
- Show how to export Isaac Sim results back to Gazebo for testing

## 5. Vision-Language-Action (VLA) System Architecture

**Decision**: Implement voice-to-action pipeline: Speech Recognition → LLM Planning → ROS 2 Action Execution

**Rationale**:
- Demonstrates how large language models interface with embodied systems
- Voice input is intuitive and engaging for learners
- Shows semantic decomposition (natural language → actionable robot commands)
- Clear end-to-end flow that ties together all modules

**Architecture**:
```
Spoken Input → Whisper/speech-recognition → Text
Text → LLM (Llama 2/Mistral) → Semantic Action Plan
Action Plan → ROS 2 Action Calls → Robot Execution
Sensor Feedback → LLM Context → Adaptive Replanning
```

**Key Components**:
1. **Speech Recognition**: Use `speech_recognition` library or Whisper for transcription
2. **LLM Planning**: Prompt-engineer an LLM to output structured actions (move, grasp, navigate)
3. **Execution Layer**: ROS 2 action servers for (navigate_to, grasp_object, place_object, etc.)
4. **Feedback Loop**: LLM-aware planning that incorporates sensor data and past actions

**Alternatives Considered**:
- **Gesture-based input**: Less intuitive; voice is more natural for humans
- **Direct neural network**: Omits interpretable symbolic planning; less teachable

**Best Practices**:
- Use few-shot prompting to guide LLM behavior without fine-tuning
- Implement graceful degradation for unexecutable plans (e.g., "fly to the moon")
- Show safety constraints (action validation) before sending to robot
- Use function calling (if supported by LLM) for structured output

## 6. Sim-to-Real Transfer Principles

**Decision**: Teach domain randomization, fine-tuning, and hardware-in-loop testing as core mitigation strategies

**Rationale**:
- Sim-to-real gap is the most critical challenge in embodied AI; must be addressed explicitly
- Domain randomization is proven technique for improving transfer
- Multiple mitigation strategies give students tools for different scenarios

**Key Concepts**:
1. **Simulation Gaps**: Dynamics discrepancy, sensor noise, timing delays, friction/contact modeling
2. **Domain Randomization**: Randomizing visual appearance, physics parameters, sensor properties
3. **Fine-tuning**: Adapting policies trained in sim to real data (few-shot learning)
4. **Hardware-in-Loop**: Testing in simulation with actual hardware controllers or sensors
5. **Evaluation Metrics**: How to measure transfer success (success rate, smoothness, robustness)

**Alternatives Considered**:
- **Omit sim-to-real**: Would leave students unprepared for real-world deployment challenges
- **Only simulation**: Without understanding transfer, students cannot extend to hardware

**Best Practices**:
- Provide concrete failure case studies showing simulation-reality discrepancies
- Show Python code for domain randomization in Gazebo and Isaac
- Include checklist for roboticists: "Is my simulation fidelity adequate for this task?"

## 7. Python Code Testing & Validation Framework

**Decision**: Use pytest for example validation; automate code example testing in CI pipeline

**Rationale**:
- pytest is standard in Python ecosystem and familiar to target audience
- Automated testing ensures all book examples work when published
- CI integration catches regressions when dependencies update

**Framework**:
- Each code example has corresponding pytest test
- Test verifies syntax, imports, and expected output
- CI workflow runs all tests on push; fails build if tests break

**Alternatives Considered**:
- **Manual testing**: Error-prone; code examples rot over time
- **No testing**: Violates spec requirement (SC-009: 95% of code examples must run without errors)

**Best Practices**:
- Use fixtures for ROS 2 node setup/teardown
- Mock external services (LLM API, Gazebo simulator) for unit tests
- Integration tests for end-to-end pipelines
- Generate test coverage reports

## 8. Capstone Architecture: Minimal Viable Autonomous Humanoid

**Decision**: Implement a modular humanoid system with 4 core components: perception, planning, control, and voice interface

**Rationale**:
- Demonstrates integration of all 4 modules
- Modular design allows students to swap/extend components
- Voice-to-action provides compelling demo and aligns with VLA module

**System Components**:

1. **Perception Module**
   - Input: Simulated camera, LiDAR from Gazebo
   - Output: Object locations, robot localization
   - Key algorithms: Visual SLAM or basic occupancy mapping

2. **Planning Module**
   - Input: Task description (from LLM), current robot state
   - Output: Sequence of ROS 2 actions (navigate, grasp, place)
   - Implements: Task decomposition, collision checking

3. **Control Module**
   - Input: High-level actions from planner
   - Output: Low-level joint commands via ROS 2 action servers
   - Implements: Inverse kinematics, trajectory planning (via MoveIt or custom)

4. **Voice Interface**
   - Input: Spoken natural language command
   - Output: Structured action plan (via LLM)
   - Implements: Speech recognition, language understanding, action validation

**Capstone Tasks** (examples students complete):
- Navigate to an object ("go to the red ball")
- Grasp and move ("pick up the object and place it on the table")
- Complex sequencing ("fetch the item from the shelf and bring it here")

**Alternatives Considered**:
- **Simpler capstone** (e.g., just navigation): Would not showcase full system integration
- **Real hardware capstone**: Out of scope per spec; simulation provides safer learning environment

**Best Practices**:
- Provide pre-built URDF humanoid model
- Pre-implement core ROS 2 nodes; students extend/customize
- Use standard algorithms (MoveIt for manipulation, Nav2 for navigation)
- Include extensive documentation and troubleshooting guides

## 9. LLM Selection & Integration

**Decision**: Default to open-source models (Llama 2, Mistral) with optional OpenAI API for advanced exercises

**Rationale**:
- Open-source models avoid licensing and API cost concerns for students
- Llama 2 has active community support and good documentation
- Mistral offers smaller models for resource-constrained setups
- OpenAI API provides cutting-edge models as optional path

**Integration Pattern**:
- Use local inference (ollama/vLLM) for reproducibility
- Simple prompt template system for action planning
- Function calling (if supported) for structured output

**Alternatives Considered**:
- **Proprietary LLMs only**: Higher cost, licensing complexity for students
- **Train custom LLM**: Out of scope per spec; use transfer learning instead

**Best Practices**:
- Document prompt engineering best practices for task decomposition
- Show how to add few-shot examples for better instruction following
- Include safety filtering (prompt validation, action whitelisting)

---

## Summary of Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Documentation Platform | Docusaurus 3.x + GitHub Pages | Free, GitHub-native, powerful |
| Primary Simulator | Gazebo 11+ | Open-source, standard, lower barrier |
| ROS 2 Language | Python (rclpy) | Accessible, sufficient for teaching |
| Teaching Approach | Progressive concepts + practical examples | Matches learner mental models |
| Capstone VLA | Voice-to-action pipeline | Intuitive, demonstrates integration |
| LLM Strategy | Open-source (Llama 2/Mistral) + optional OpenAI | Balance cost, accessibility, performance |
| Sim-to-Real | Domain randomization + fine-tuning | Proven techniques, teachable |
| Code Testing | pytest + CI automation | Ensures example quality |

## Open Questions Resolved

✓ **Docusaurus vs. alternatives**: Docusaurus chosen for power + ease
✓ **ROS 2 teaching path**: Progressive layer approach (nodes → topics → services → actions)
✓ **Simulation platform**: Gazebo primary + Isaac as advanced topic
✓ **Capstone scope**: Modular voice-to-action system integrating all 4 modules
✓ **LLM selection**: Open-source default, OpenAI optional
✓ **Sim-to-real strategy**: Domain randomization + fine-tuning + hardware-in-loop concepts

## Next Steps (Phase 1)

1. Create `data-model.md` defining content structure and module breakdown
2. Draft Docusaurus contracts: site navigation, configuration schema
3. Define capstone API contracts: ROS 2 node interfaces, message schemas
4. Create code example validation contracts
5. Write `quickstart.md` for reader/developer setup
