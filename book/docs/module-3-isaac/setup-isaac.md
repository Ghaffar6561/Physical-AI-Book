# Setting Up NVIDIA Isaac Sim

This guide covers installing NVIDIA Isaac Sim and configuring it for the Physical AI book exercises. Isaac Sim is the industry-standard photorealistic robot simulator used by companies like Boston Dynamics, Tesla, and major AI labs.

---

## Prerequisites

Before starting, ensure you have:

- **NVIDIA GPU** (RTX 3060 or better recommended)
  - RTX 3060: ~100 parallel environments, 10K FPS
  - RTX 3090: ~1000 parallel environments, 100K FPS
  - A100: 10,000+ parallel environments (for research labs)
- **CUDA 11.4+** (if GPU is supported)
- **8GB+ VRAM** (16GB+ recommended for synthetic data generation)
- **20GB+ disk space** for Isaac Sim installation
- **Ubuntu 18.04+, Windows 10+, or macOS 10.15+**
- Python 3.7+

---

## Installation

### Option 1: Docker (Recommended for Reproducibility)

Docker ensures identical environment across machines, ideal for development and education.

**Step 1: Install Docker**

```bash
# Ubuntu/Debian
sudo apt-get install docker.io docker-compose

# Verify installation
docker --version
```

**Step 2: Get Isaac Docker Image**

```bash
# Pull official Isaac Gym image
docker pull nvcr.io/nvidia/isaac-gym:latest

# Or for Isaac Sim (larger, more complete)
docker pull nvcr.io/nvidia/isaac-sim:2022.2.0
```

**Step 3: Run Isaac Container**

```bash
# For GPU support, use nvidia-docker
sudo apt-get install nvidia-docker2

# Run container with GPU access
sudo nvidia-docker run --gpus all -it \
  --volume $(pwd)/workspace:/workspace \
  nvcr.io/nvidia/isaac-gym:latest

# Inside container
source /opt/miniconda3/bin/activate

# Verify Isaac Gym
python -c "from isaacgym import gymapi; print('Isaac Gym ready!')"
```

**Step 4: Mount Local Code**

```bash
# Create workspace directory
mkdir -p ~/isaac_workspace/code

# Run container with code mounted
nvidia-docker run --gpus all -it \
  --volume ~/isaac_workspace/code:/workspace/code \
  --volume ~/isaac_workspace/data:/workspace/data \
  nvcr.io/nvidia/isaac-gym:latest
```

### Option 2: Native Installation (Ubuntu 20.04+)

For direct integration with existing development environment.

**Step 1: Install CUDA Toolkit**

```bash
# Check if CUDA already installed
nvidia-smi

# If not, install CUDA 11.4
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda_11.4.0_470.42.01_linux.run

sudo sh cuda_11.4.0_470.42.01_linux.run
```

**Step 2: Install Isaac Gym via Pip**

```bash
# Create virtual environment
python3 -m venv isaac_env
source isaac_env/bin/activate

# Install Isaac Gym
pip install isaacgym

# Install common dependencies
pip install torch numpy matplotlib

# Verify
python -c "from isaacgym import gymapi; print('Isaac Gym installed!')"
```

**Step 3: Install Isaac Sim (Optional, for Graphics)**

```bash
# Download from NVIDIA Omniverse
# https://www.nvidia.com/en-us/omniverse/download/

# Or install via apt (Ubuntu 20.04+)
sudo apt-get install nvidia-isaac-sim

# Verify
isaacSim --help
```

### Option 3: Omniverse Launcher (Full IDE)

For complete Isaac Sim with visual editor.

**Step 1: Download Omniverse Launcher**

```bash
# Visit: https://www.nvidia.com/en-us/omniverse/
# Download launcher for your OS
# Install and open launcher
```

**Step 2: Install Isaac Sim via Launcher**

- Open Omniverse Launcher
- Navigate to "Apps"
- Find "Isaac Sim"
- Click "Install"
- Wait ~30 minutes for download/install

**Step 3: Launch Isaac Sim**

```bash
# From launcher, click "Launch" on Isaac Sim
# Or from command line
~/.local/share/ov/pkg/isaac-sim-2022.2.0/isaac-sim.sh
```

---

## Configuration

### Set Up Isaac Environment

Create `~/.bashrc` alias for quick access:

```bash
# Add to ~/.bashrc
alias isaac_activate='source ~/isaac_env/bin/activate && export PYTHONPATH=/workspace/code:$PYTHONPATH'
alias isaac_docker='nvidia-docker run --gpus all -it -v ~/isaac_workspace:/workspace nvcr.io/nvidia/isaac-gym:latest'

# Reload
source ~/.bashrc
```

### Configure NVIDIA Settings (Optional)

Optimize GPU for Isaac:

```bash
# Enable GPU persistence mode (faster launches)
sudo nvidia-smi -pm 1

# Set GPU power limit for consistent performance
sudo nvidia-smi -pl 280  # RTX 3090 (adjust for your GPU)

# Check current settings
nvidia-smi -q -d clock
```

---

## Testing Installation

### Test 1: Basic Isaac Gym

```python
# test_isaac_installation.py
from isaacgym import gymapi
from isaacgym import gymutil

# Create gym
gym = gymapi.create_gym()

# Create simulation
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = [0.0, 0.0, -9.81]

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create environment
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

# Success!
print("✓ Isaac Gym installation verified")
```

**Run test:**

```bash
python test_isaac_installation.py
```

**Expected output:** `✓ Isaac Gym installation verified`

### Test 2: Simple Environment

```python
# test_simple_task.py
from isaacgym import gymapi
import numpy as np

gym = gymapi.create_gym()

# Create simulation
sim_params = gymapi.SimParams()
sim_params.use_gpu_pipeline = True
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create 10 parallel environments
num_envs = 10
envs = []
for i in range(num_envs):
    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
    envs.append(env)

print(f"✓ Created {num_envs} parallel environments")

# Simulate 100 steps
gym.prepare_sim(sim)
for step in range(100):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

print("✓ Physics simulation working")
```

**Run test:**

```bash
python test_simple_task.py
```

**Expected output:** Both ✓ messages

### Test 3: Synthetic Data Generation

```python
# test_synthetic_data.py
from isaacgym import gymapi
import numpy as np

gym = gymapi.create_gym()

# Create simulation
sim_params = gymapi.SimParams()
sim_params.use_gpu_pipeline = True
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create environment with camera
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)

# Create camera sensor
camera = gym.create_camera_sensor(sim)

# Render
gym.prepare_sim(sim)
gym.render(sim)

# Capture image
image_data = gym.get_camera_image_gpu_tensor(
    sim, env, camera, gymapi.IMAGE_COLOR
)

print(f"✓ Captured image: shape={image_data.shape}")
```

---

## Troubleshooting

### Issue 1: CUDA Not Found

```
ImportError: libcuda.so.1: cannot open shared object file
```

**Solution:**

```bash
# Check NVIDIA drivers
nvidia-smi

# Install CUDA if missing
sudo apt-get install cuda-11-4

# Add to PATH
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
```

### Issue 2: GPU Out of Memory

```
RuntimeError: GPU out of memory
```

**Solution:**

- Reduce number of parallel environments
- Use FP16 precision instead of FP32
- Monitor GPU with `nvidia-smi dmon`

```python
# Reduce memory usage
torch.set_default_dtype(torch.float16)  # Use half precision
num_envs = min(100, gpu_memory_gb * 50)  # Conservative estimate
```

### Issue 3: Docker GPU Not Recognized

```
docker: Error response from daemon: could not select device driver
```

**Solution:**

```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test
nvidia-docker run --rm --gpus all nvidia/cuda:11.4.0-runtime-ubuntu20.04 nvidia-smi
```

### Issue 4: Slow Rendering (< 1000 FPS)

**Solution:**

- Enable GPU-accelerated rendering: `sim_params.use_gpu_pipeline = True`
- Reduce camera resolution
- Disable visualizations if not needed
- Check thermal throttling: `nvidia-smi -q -d temperature`

---

## Development Workflow

### Setup Project Structure

```bash
# Create project directory
mkdir -p isaac_humanoid_project/{sim,code,results,data}

# Structure
isaac_humanoid_project/
├── sim/                    # Isaac simulation configs
│   ├── robot.urdf
│   ├── world.sdf
│   └── assets/
├── code/                   # Python code
│   ├── train.py           # RL training loop
│   ├── eval.py            # Evaluation script
│   └── utils/             # Helper modules
├── results/               # Training outputs
│   ├── checkpoints/
│   ├── logs/
│   └── videos/
└── data/                  # Datasets
    ├── synthetic/
    └── real_robot/
```

### Typical Training Workflow

```bash
# 1. Launch Isaac environment (Docker or local)
isaac_activate

# 2. Run training
cd ~/isaac_humanoid_project
python code/train.py --num-envs 1000 --epochs 100

# 3. Monitor GPU
nvidia-smi dmon

# 4. Evaluate policy
python code/eval.py --checkpoint results/checkpoints/best.pt

# 5. Generate synthetic data
python code/gen_synthetic_data.py --output-dir data/synthetic/

# 6. Analyze results
jupyter notebook results/analysis.ipynb
```

---

## Performance Tuning

### Profile Your Code

```python
# Profile Isaac operations
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your Isaac code here
for step in range(1000):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Benchmark Configurations

```
Configuration          Environments    FPS      Memory
─────────────────────────────────────────────────────
RTX 3060 baseline      100            10,000    6GB
RTX 3060 optimized     200            15,000    8GB
RTX 3090 baseline      1000           100,000   16GB
RTX 3090 max           10,000         800,000   24GB
A100 (research)        50,000         5M        80GB
```

---

## Free Tier & Academic Access

### NVIDIA Isaac Free Community Edition

- Unlimited usage for non-commercial research
- Full Isaac Gym & Isaac Sim functionality
- Free Newton supercomputing credits (if accepted)
- Community support via forums

**Register:** https://www.nvidia.com/en-us/omniverse/

### Academic GPU Access

If you don't have a local GPU:

1. **Google Colab** (Free)
   ```bash
   # Limited NVIDIA T4 GPU access
   # 12 hours per session
   # Not recommended for long Isaac training
   ```

2. **NVIDIA GRID Cloud** (Free tier available)
   - Remote GPU rental for students
   - Apply at: https://www.nvidia.com/en-us/ai-data-science/gpu-cloud/

3. **University HPC** (If available)
   - Many universities provide GPU clusters
   - Contact your IT/HPC department

4. **AWS/GCP/Azure Credits**
   - Startup program credits
   - Research grants often include cloud credits

---

## Next Steps

1. **Verify installation** using the three tests above
2. **Run example from course**: `code/domain_randomization.py`
3. **Try synthetic data generation**: Follow isaac-workflows.md code example
4. **Train your first policy**: Adapt the GraspingTask template from isaac-workflows.md
5. **Compare with Gazebo**: Run same task in both simulators, compare performance

---

## Resources

### Official Documentation
- **Isaac Gym Docs**: https://docs.omniverse.nvidia.com/isaacsim/
- **Isaac Gym Examples**: https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
- **PhysX Documentation**: https://github.com/NVIDIAGameWorks/PhysX

### Tutorials & Courses
- **NVIDIA Isaac Sim Tutorials**: https://www.nvidia.com/en-us/omniverse/learning-hub/
- **RL Training with Isaac**: https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_gym_RL_example.html
- **Synthetic Data Generation**: https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_replicator_getting_started.html

### Community
- **NVIDIA Forums**: https://forums.developer.nvidia.com/c/ai-data-science/isaac-platform/
- **GitHub Discussions**: https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/discussions
- **Discord Community**: Join NVIDIA Omniverse Discord

---

## Key Differences: Isaac vs Gazebo

| Feature | Gazebo | Isaac Sim |
|---------|--------|-----------|
| **Physics** | ODE, Bullet | PhysX (same as AAA games) |
| **Rendering** | Basic OpenGL | Photorealistic RTX ray-tracing |
| **Data Gen** | Manual | Automatic with synthetic labels |
| **RL Speed** | 1× (baseline) | 100-1000× faster |
| **Vision Tasks** | 60% transfer | 95% transfer |
| **Ease** | Easier | Steeper curve |
| **GPU Required** | No | Yes (RTX 3060+) |
| **Cost** | Free | Free (community), paid (enterprise) |

---

## Conclusion

You're now ready to:

✅ Install Isaac Sim in Docker or natively
✅ Set up development environment
✅ Run physics simulations with 1000s of parallel environments
✅ Generate synthetic training data automatically
✅ Train RL policies 100× faster than Gazebo

**Next**: Follow the isaac-workflows.md code examples to train your first vision-based grasping policy!
