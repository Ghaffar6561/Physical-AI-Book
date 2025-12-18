# NVIDIA Isaac: Photorealistic Simulation & Synthetic Data Generation

You've learned how to bridge sim-to-real transfer using domain randomization. But some tasks—vision-intensive tasks, complex manipulation—need photorealistic rendering. This chapter covers NVIDIA Isaac, the industry-standard platform for photorealistic robot simulation.

---

## When Do You Need Isaac?

### Gazebo (What You've Used) vs Isaac (Advanced)

**Gazebo:**
- Physics: ODE or Bullet engine (good, fast)
- Rendering: Simple geometric shapes, basic textures
- Graphics: Real-time but non-photorealistic
- Cost: Free, open-source
- Learning curve: Moderate
- Real-world fidelity: 70% (good for control tasks)

**NVIDIA Isaac:**
- Physics: PhysX (same engine as AAA games)
- Rendering: Physically-based rendering (PBR), ray-tracing
- Graphics: Photorealistic, indistinguishable from real photos
- Cost: Free (community), paid for enterprise
- Learning curve: Steep (NVIDIA ecosystem)
- Real-world fidelity: 95%+ (excellent for vision tasks)

### Decision Matrix: When to Use Each

```
Task Type           Gazebo      Isaac
─────────────────────────────────────────
Sim-to-real (control)    ✓ Sufficient
Sim-to-real (vision)               ✓ Needed
Robot assembly            ✓ OK
Fine manipulation                  ✓ Better
Grasping (force-based)    ✓ Good
Grasping (vision-based)            ✓ Best
Bipedal walking       ✓ Sufficient
Humanoid soccer                    ✓ Needed
Autonomous vehicles  ✓ Sufficient
Self-driving (ML)                  ✓ Needed
```

**Simple rule:**
- If training **control policies**: Use Gazebo (fast iteration)
- If training **perception models**: Use Isaac (photorealistic data)
- If budget permits: Use Isaac for both (best results)

---

## NVIDIA Isaac: Architecture

### Components

Isaac is a modular ecosystem:

```
NVIDIA Isaac SDK (C++ core)
├─ Isaac Gym (RL training environment)
├─ Isaac Sim (Simulation engine)
│  ├─ PhysX physics
│  ├─ RTX ray-tracing renderer
│  └─ Synthetic data generation
├─ Isaac Nav2 (Navigation stack)
└─ Isaac Perceptor (Vision modules)
```

**For this course, we focus on:**
1. **Isaac Sim** — Photorealistic simulation
2. **Synthetic data generation** — Training vision models
3. **Isaac Gym** — RL policy training

### Key Features

**Photorealistic Rendering:**
```
Ray-tracing with RTX cores (NVIDIA GPUs)
  - Per-pixel ray casting (not rasterization)
  - Physically accurate reflection/refraction
  - Global illumination (bouncing light)
  - Shadows, ambient occlusion
  - Material properties (roughness, metallic, IOR)

Result: Images virtually identical to real camera photos
```

**Physics Engine (PhysX):**
```
Same engine used in AAA games (Unreal, Unity)
  - Accurate contact dynamics
  - Friction models (kinetic + static)
  - Joint constraints (motors with saturation)
  - Soft-body simulation
  - Fluid dynamics (for grasping wet objects)

Result: Physics matches real world better than Gazebo
```

**Synthetic Data Generation:**
```
Automatic labeling + variation:
  - Generate diverse images with known ground truth
  - Randomize lighting, camera viewpoint, object poses
  - Export bounding boxes, segmentation masks, depth
  - Generate 10,000s of synthetic training images in hours

Result: Train perception models without manual labeling
```

---

## Workflow 1: Training Vision-Based Grasping with Isaac

### Typical Pipeline

```
Step 1: Design scene in Isaac Sim
  ├─ Load humanoid robot URDF
  ├─ Add objects (bottles, boxes, etc.)
  ├─ Configure camera sensor
  └─ Set up lighting

Step 2: Configure synthetic data generation
  ├─ Randomization: Object poses, materials, lighting
  ├─ Export format: RGB images, depth, segmentation
  └─ Quality: 4K photorealistic renders

Step 3: Generate large synthetic dataset
  ├─ Run 100,000 randomized renders
  ├─ Automatic ground truth labels (CNN training labels)
  ├─ Takes ~2-4 hours on single GPU
  └─ Results: 100K labeled images

Step 4: Train perception CNN on synthetic data
  ├─ Standard PyTorch/TensorFlow training
  ├─ Dataset: 100K synthetic images + labels
  ├─ Model: ResNet-50 object detector
  └─ Accuracy: 95% on synthetic test set

Step 5: Domain adaptation (fine-tune on real images)
  ├─ Collect 100 real images from robot camera
  ├─ Fine-tune last few layers (10-20 epochs)
  └─ Real accuracy: 92% (slight drop due to domain shift)

Step 6: Deploy to robot
  ├─ Run gripper controller with vision
  ├─ Real-world success rate: 90%+
  └─ Comparable to hand-crafted solutions
```

### Code Example: Isaac Synthetic Data Generation

```python
"""
Synthetic data generation in NVIDIA Isaac Sim.

Generates diverse images of grasping scenarios:
  - Randomized object positions
  - Randomized lighting directions
  - Randomized camera views
  - Automatic ground truth labels
"""

from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np


class SyntheticDataGenerator:
    """Generate synthetic grasping training data."""

    def __init__(self, num_envs=8):
        """Initialize Isaac environment."""
        self.gym = gymapi.create_gym()

        # Create simulation
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = np.array([0.0, 0.0, -9.81])
        sim_params.dt = 1.0 / 60.0  # 60 Hz physics
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True

        self.sim = self.gym.create_sim(
            compute_device_id=0,
            graphics_device_id=0,
            type=gymapi.SIM_PHYSX,
            params=sim_params
        )

        # Create camera
        self.camera = self.gym.create_camera_sensor(self.sim)

        # Load robot and objects
        self.robot_asset = self.gym.load_asset(
            self.sim, "path/to/humanoid_simple.urdf"
        )
        self.bottle_asset = self.gym.load_asset(
            self.sim, "path/to/bottle.urdf"
        )

        self.num_envs = num_envs
        self.envs = []
        self.actors = []

        # Create parallel environments
        for i in range(num_envs):
            self._create_env(i)

    def _create_env(self, env_id):
        """Create a single environment with robot and object."""
        env = self.gym.create_env(
            self.sim,
            lower=gymapi.Vec3(-2, -2, 0),
            upper=gymapi.Vec3(2, 2, 2),
            num_per_row=4
        )

        # Add robot
        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0, 0, 0)
        robot_actor = self.gym.create_actor(
            env, self.robot_asset, robot_pose, "robot", 0, 0
        )

        # Add object (bottle)
        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(
            np.random.uniform(0.3, 0.7),  # x: randomized
            np.random.uniform(-0.2, 0.2),  # y: randomized
            0.1  # z: on table
        )
        object_actor = self.gym.create_actor(
            env, self.bottle_asset, object_pose, "bottle", 0, 0
        )

        self.envs.append(env)
        self.actors.append((robot_actor, object_actor))

    def randomize_scene(self):
        """Randomize lighting, materials, and object positions."""
        for env_id in range(self.num_envs):
            env = self.envs[env_id]
            _, obj_actor = self.actors[env_id]

            # Randomize object position
            new_pose = gymapi.Transform()
            new_pose.p = gymapi.Vec3(
                np.random.uniform(0.3, 0.7),
                np.random.uniform(-0.2, 0.2),
                0.1
            )
            self.gym.set_actor_transform(env, obj_actor, new_pose)

            # Randomize lighting direction
            light_direction = np.random.uniform(-1, 1, 3)
            light_direction = light_direction / np.linalg.norm(light_direction)
            # (Note: In real code, set light via gym API)

            # Randomize material properties (in real code)
            # friction, roughness, color

    def capture_data(self):
        """Capture RGB, depth, and segmentation images."""
        self.gym.prepare_sim(self.sim)

        # Render scene
        self.gym.render(self.sim)

        # Capture images from all environments
        images = []
        depths = []
        segmentation_masks = []

        for env_id in range(self.num_envs):
            # Capture camera data
            image_data = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[env_id], self.camera, gymapi.IMAGE_COLOR
            )
            depth_data = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[env_id], self.camera, gymapi.IMAGE_DEPTH
            )
            segmentation_data = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[env_id], self.camera, gymapi.IMAGE_SEGMENTATION
            )

            images.append(image_data)
            depths.append(depth_data)
            segmentation_masks.append(segmentation_data)

        return images, depths, segmentation_masks

    def generate_dataset(self, num_images=10000):
        """Generate synthetic dataset."""
        dataset = {
            'images': [],
            'depths': [],
            'masks': [],
            'poses': []  # Ground truth object poses
        }

        for step in range(num_images // self.num_envs):
            # Randomize all parameters
            self.randomize_scene()

            # Capture current frame
            images, depths, masks = self.capture_data()

            # Get ground truth object positions
            for env_id in range(self.num_envs):
                _, obj_actor = self.actors[env_id]
                pose = self.gym.get_actor_transform(
                    self.envs[env_id], obj_actor
                )

                dataset['images'].append(images[env_id])
                dataset['depths'].append(depths[env_id])
                dataset['masks'].append(masks[env_id])
                dataset['poses'].append((pose.p.x, pose.p.y, pose.p.z))

            # Log progress
            if step % 100 == 0:
                print(f"Generated {step * self.num_envs}/{num_images} images")

        return dataset


def main():
    """Generate 10,000 synthetic training images."""
    generator = SyntheticDataGenerator(num_envs=8)

    # Generate dataset
    dataset = generator.generate_dataset(num_images=10000)

    # Save to disk
    import pickle
    with open('synthetic_grasping_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Saved {len(dataset['images'])} training images")

    # Train CNN on this dataset (use PyTorch)
    # model = train_cnn(dataset['images'], dataset['poses'])


if __name__ == '__main__':
    main()
```

### Key Insights

1. **Parallel environments**: Isaac Sim runs 8+ environments simultaneously on one GPU
   - Standard Gazebo: 1 environment per instance
   - Isaac: 8 environments on single GPU
   - Result: 8× faster data generation

2. **Automatic ground truth**: Objects automatically labeled with poses
   - No manual annotation needed
   - Perfect labels (no human error)
   - Scales to 100,000s of images

3. **Domain adaptation required**: Synthetic → real still has gap
   - Even photorealistic Isaac differs from real cameras (slight color shift, minor sensor noise patterns)
   - Fine-tuning on 100 real images recovers most performance
   - Typical synthetic → real accuracy drop: 95% → 92%

---

## Workflow 2: Isaac Gym for Reinforcement Learning

For training **control policies** with photorealistic physics.

### Isaac Gym Architecture

```
Isaac Gym (GPU-accelerated RL)
├─ Parallel environments (1000s running simultaneously)
├─ GPU-based physics simulation (PhysX)
├─ PyTorch integration (native tensor operations)
└─ PPO/SAC/other RL algorithms
```

### Example: Training Grasping Policy

```python
"""
Isaac Gym PPO training for robotic grasping.

Trains a humanoid to grasp objects using reinforcement learning.
Runs 1000 parallel environments on single GPU.
"""

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *
import torch


class GraspingTask:
    """RL task for learning grasping."""

    def __init__(self, device="cuda:0"):
        self.device = device
        self.gym = gymapi.create_gym()

        # Create simulation
        sim_params = gymapi.SimParams()
        sim_params.use_gpu_pipeline = True
        self.sim = self.gym.create_sim(
            compute_device_id=0,
            graphics_device_id=0,
            type=gymapi.SIM_PHYSX,
            params=sim_params
        )

        # Create 1000 parallel environments
        self.num_envs = 1000
        self.envs = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower=gymapi.Vec3(-1, -1, 0), upper=gymapi.Vec3(1, 1, 2), num_per_row=32)
            self.envs.append(env)

    def reset(self):
        """Reset all environments to initial state."""
        # Randomize object positions for each environment
        for env_id in range(self.num_envs):
            # Set robot to home position
            # Set object to random position
            pass

    def step(self, actions):
        """
        Execute actions in all environments.

        Args:
            actions: Tensor of shape (num_envs, num_joints)

        Returns:
            observations: Tensor of shape (num_envs, obs_dim)
            rewards: Tensor of shape (num_envs,)
            dones: Tensor of shape (num_envs,)
        """
        # Set joint targets
        # Execute physics step
        # Compute rewards
        # Check termination conditions
        pass

    def compute_reward(self, env_ids):
        """Reward function: +1 if object grasped, -0.01 * action magnitude."""
        # Distance to object
        # Gripper force
        # Object position change
        pass


# Training loop
task = GraspingTask()

for epoch in range(100):
    for step in range(1000):
        # Generate actions from policy
        actions = policy(observations)

        # Step all 1000 environments simultaneously
        obs, rewards, dones = task.step(actions)

        # Collect trajectories
        memory.store(obs, actions, rewards)

    # PPO update
    policy.update(memory)

    # Log metrics
    print(f"Epoch {epoch}: Mean reward {rewards.mean():.2f}")
```

### Performance Scaling

```
Single-GPU performance with Isaac Gym:

GPU             Environments    Physics FPS    Total FPS
─────────────────────────────────────────────────────────
RTX 3060        100             1000          10,000
RTX 3090        1000            1000          100,000
RTX A100 (80GB) 10000           1000          1,000,000

For comparison, Gazebo on CPU:
  1 environment at 50 Hz = 50 FPS
  Isaac: 10,000× faster
```

---

## Comparing Simulation Fidelity

### Rendering Quality Spectrum

```
Fidelity Level     Engine          Rendering Quality         Vision Tasks
────────────────────────────────────────────────────────────────────────
Low                Gazebo          Simple geometry           Not suitable
                   (fast, real-time)
Medium             Gazebo+textures  Textured models           Marginal
                   (real-time)
High               Isaac Sim        Photorealistic            Good (with adaptation)
                   (near-real-time)
Very High          Isaac RTX        Photorealistic + ray-trace Excellent
                   (real-time with GPU)
```

### Accuracy Comparison: Grasping CNN Training

```
Training Source     Train Accuracy   Real Accuracy   Transfer Success
─────────────────────────────────────────────────────────────────────
Gazebo renders      85%              30%             Fails ✗
Gazebo + noise      90%              50%             Poor
Isaac synthetic     95%              85%             Good ✓
Isaac + real FT     95%              92%             Excellent ✓✓
```

---

## Photorealistic vs Physics-Accurate: Trade-offs

### What Matters for Different Tasks

**For Control Tasks (force-based grasping):**
- Physics accuracy: CRITICAL
- Rendering quality: Not important
- Recommendation: **Gazebo** (faster iteration, physics sufficient)

**For Vision Tasks (vision-based grasping):**
- Physics accuracy: Important
- Rendering quality: CRITICAL
- Recommendation: **Isaac Sim** (photorealistic + good physics)

**For Manipulation in Complex Scenes:**
- Physics accuracy: CRITICAL
- Rendering quality: CRITICAL
- Recommendation: **Isaac Sim** (both excellent)

---

## Practical Tips: Using Isaac for Your Project

### 1. Installation

```bash
# Option A: Docker (recommended for reproducibility)
docker pull nvcr.io/nvidia/isaac-gym:latest
docker run --gpus all -it nvcr.io/nvidia/isaac-gym:latest

# Option B: Native (requires CUDA 11.4+)
pip install isaacgym
pip install isaacgymenvs

# Verify
python -c "from isaacgym import gymapi; print('Isaac Gym installed!')"
```

### 2. Learning Resources

```
Official documentation:
  https://docs.omniverse.nvidia.com/isaacsim/

Key tutorials:
  1. Getting started with Isaac Gym
  2. Physics simulation setup
  3. Synthetic data generation
  4. RL training with PPO

Community:
  NVIDIA Forums: forums.developer.nvidia.com
  GitHub: github.com/NVIDIA-Omniverse/IsaacGymEnvs
```

### 3. Performance Optimization

**GPU memory management:**
```python
# Use FP16 precision to reduce memory
torch.set_default_dtype(torch.float16)

# Batch rendering (efficient multi-env)
num_envs = min(gpu_memory_gb * 100, 10000)  # Rough rule
```

**Physics stepping:**
```python
# Balance speed vs accuracy
sim_params.substeps = 2  # 2-4 recommended
sim_params.dt = 1/60  # 60 Hz is typical
```

---

## When NOT to Use Isaac

### Situations Where Gazebo is Better

1. **Quick prototyping**: Isaac has more overhead, slower iteration
2. **CPU-only systems**: Gazebo works on CPU, Isaac requires GPU
3. **Control-heavy tasks**: Physics fidelity more important than rendering
4. **Learning/education**: Gazebo easier to understand and debug
5. **Legacy systems**: Existing code in Gazebo ecosystem

---

## Synthetic Data Generation Best Practices

### Quality Checklist

- [ ] Randomize object positions (not just location, also orientation)
- [ ] Randomize lighting (direction, intensity, color temperature)
- [ ] Randomize camera viewpoint (5-10 different angles per scenario)
- [ ] Randomize materials (friction, roughness, reflectivity)
- [ ] Randomize object appearance (colors, textures)
- [ ] Randomize scene clutter (add other objects for occlusion)
- [ ] Export ground truth labels (bounding boxes, masks, 6D poses)
- [ ] Generate 10,000+ images (more = better for deep learning)
- [ ] Balance classes (if some objects rarer, oversample)
- [ ] Validate on real images before deployment

### Typical Workflow for 100K Image Dataset

```
Step 1: Design scene (1 hour)
Step 2: Set up randomization (2 hours)
Step 3: Generate 100K images (4 hours on RTX 3090)
Step 4: Train CNN (2-8 hours GPU time)
Step 5: Fine-tune on real images (1 hour)
Step 6: Validate (30 min)

Total: ~15 hours, mostly waiting for rendering
```

---

## Key Takeaways

| Aspect | Gazebo | Isaac Sim |
|--------|--------|-----------|
| **Physics** | Good (ODE) | Excellent (PhysX) |
| **Rendering** | Basic | Photorealistic |
| **Synthetic data** | Manual | Automatic |
| **RL training speed** | 1× (baseline) | 10-100× |
| **Vision task fidelity** | 60% | 95% |
| **Cost** | Free | Free (+ paid options) |
| **Learning curve** | Moderate | Steep |
| **Best for** | Control + quick iteration | Vision + production |

---

## The Bottom Line

**Gazebo is sufficient for 70% of robotics projects.**

Use Isaac when:
1. Training vision-based perception models
2. Domain randomization on Gazebo insufficient
3. You have GPU budget and GPU hardware
4. Photorealism directly impacts task success
5. Deploying to demanding real-world scenarios

For the Physical AI book, we've focused on Gazebo (accessible, educational) but understand that **production systems typically use Isaac Sim** for perception-heavy tasks.

---

## Further Reading

- **NVIDIA Isaac Docs**: https://docs.omniverse.nvidia.com/isaacsim/
- **"Learning Dexterous In-Hand Manipulation" (Openai 2018)**: Landmark synthetic data generation for robot manipulation
- **"PhysX 4: Physics Engine Architecture"**: Understanding the physics substrate
- **Synthetic-to-Real Transfer Reviews**: Latest on domain adaptation techniques

---

**Next**: [Perception Exercises](exercises.md) — Apply sensor fusion, sim-to-real transfer, and Isaac concepts to real robotics challenges.

You now understand the full spectrum of robot simulation: from lightweight Gazebo for control tasks to photorealistic Isaac for vision-intensive applications. Choose the right tool for your problem.
