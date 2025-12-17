# Setting Up Gazebo for Humanoid Robotics

This guide covers installing Gazebo 11+, configuring ROS 2 integration, and setting up the development environment for the Physical AI book exercises.

## Prerequisites

Before starting, ensure you have:
- **Ubuntu 20.04 LTS or 22.04 LTS** (recommended)
- **ROS 2 Humble** installed ([official guide](https://docs.ros.org/en/humble/Installation.html))
- **Python 3.8+**
- **Git**
- **~5 GB disk space** for Gazebo + dependencies

If using WSL2 (Windows Subsystem for Linux):
- WSL2 with Ubuntu 22.04
- X11 server for GUI (e.g., VcXsrv, Xming)
- Virtual GPU acceleration (optional but recommended)

## Installation

### Option 1: Ubuntu/Debian (Recommended)

**Step 1: Install Gazebo 11**

```bash
# Update package lists
sudo apt-get update

# Install Gazebo
sudo apt-get install gazebo11 libgazebo11-dev

# Verify installation
gazebo --version
# Should output: Gazebo multi-robot simulator, version 11.x.x
```

**Step 2: Install ROS 2 Gazebo Bridge**

```bash
# Install gazebo_ros packages
sudo apt-get install ros-humble-gazebo-ros
sudo apt-get install ros-humble-gazebo-ros-pkgs

# Verify ROS 2 integration
ros2 pkg list | grep gazebo
# Should list: gazebo_msgs, gazebo_ros, etc.
```

**Step 3: Install Additional Robotics Tools**

```bash
# For URDF visualization and validation
sudo apt-get install liburdf-dev
sudo apt-get install ros-humble-urdf
sudo apt-get install ros-humble-urdf-tools

# For RViz visualization
sudo apt-get install ros-humble-rviz2

# For ROS 2 CLI tools
sudo apt-get install ros-humble-ros2cli
```

**Step 4: Verify Installation**

```bash
# Test Gazebo
gazebo &

# Test ROS 2 Gazebo integration
source /opt/ros/humble/setup.bash
ros2 pkg find gazebo_ros
```

### Option 2: WSL2 (Windows)

**Step 1: Set up WSL2 environment**

```bash
# In your WSL2 terminal
wsl --set-default-version 2

# Install Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# Launch and update Ubuntu
wsl -d Ubuntu-22.04
sudo apt-get update
sudo apt-get upgrade -y
```

**Step 2: Install X11 Server**

On Windows (host), download and install:
- **VcXsrv** (free) OR
- **Xming** (free) OR
- **MobaXterm** (paid, all-in-one)

**Step 3: Configure X11 in WSL2**

Add to your `~/.bashrc`:

```bash
# For WSL2 X11 forwarding
export DISPLAY=$(grep -m 1 nameserver /etc/resolv.conf | awk '{print $2}'):0.0
export LIBGL_ALWAYS_INDIRECT=1
```

Reload:
```bash
source ~/.bashrc
```

**Step 4: Install Gazebo and ROS 2**

Follow the same steps as Option 1 above.

**Step 5: Test GUI**

```bash
# Start X11 server on Windows
# Then in WSL2 terminal:
gazebo &
```

If Gazebo GUI doesn't appear:
- Check X11 server is running on Windows
- Run `echo $DISPLAY` to verify (should show `:0.0` or similar)
- Check firewall: X11 server needs to accept WSL2 connections

### Option 3: Docker (All Platforms)

For reproducible, isolated environments:

**Create a Dockerfile:**

```dockerfile
FROM osrf/ros:humble-desktop-full

# Install Gazebo and tools
RUN apt-get update && apt-get install -y \
    gazebo11 \
    libgazebo11-dev \
    ros-humble-gazebo-ros \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-urdf \
    ros-humble-rviz2 \
    python3-pip

# Install Python dependencies
RUN pip3 install pytest rclpy

# Source ROS 2
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
```

**Build and run:**

```bash
# Build image
docker build -t humanoid-sim .

# Run container with GUI support
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  humanoid-sim

# Inside container:
gazebo &
```

## Configuration

### ROS 2 Setup

Add to your shell startup file (`~/.bashrc` or `~/.zshrc`):

```bash
# ROS 2 environment
source /opt/ros/humble/setup.bash

# Gazebo environment variables
export GAZEBO_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:$GAZEBO_PLUGIN_PATH
export GAZEBO_MODEL_PATH=~/.local/share/gazebo-11/models:$GAZEBO_MODEL_PATH
export GAZEBO_RESOURCE_PATH=~/.local/share/gazebo-11:$GAZEBO_RESOURCE_PATH

# Python path for local packages
export PYTHONPATH=${PYTHONPATH}:~/PhysicalAI-Book

# Alias for quick launches
alias ros_setup='source /opt/ros/humble/setup.bash && export GAZEBO_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:$GAZEBO_PLUGIN_PATH'
```

Reload:
```bash
source ~/.bashrc
```

### Gazebo Configuration

Create `~/.gazebo/gui.conf`:

```xml
<?xml version="1.0"?>
<gui>
  <camera>
    <pose>5 -5 3 0 0.4 2.4</pose>
  </camera>
  <plugin filename="libGUISystem.so" name="gui_system"/>
</gui>
```

## Project Setup

### Clone the Physical AI Book Repository

```bash
cd ~
git clone https://github.com/your-org/PhysicalAI-Book.git
cd PhysicalAI-Book
```

### Install Python Dependencies

```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r book/examples/requirements.txt

# For development/testing:
pip install pytest pytest-cov pytest-mock
```

### Verify URDF Files

```bash
# Validate the humanoid URDF
check_urdf book/examples/humanoid-sim/gazebo_models/humanoid_simple.urdf

# Should output: robot name is: humanoid_simple
#                Checks passed.

# Visualize URDF structure
urdf_to_graphiz book/examples/humanoid-sim/gazebo_models/humanoid_simple.urdf
dot -Tpng humanoid_simple.svg -o humanoid_simple.png  # Generate image
```

## Running Your First Simulation

### Launch Gazebo with the World

```bash
cd ~/PhysicalAI-Book

# Start Gazebo with the humanoid world
gazebo --verbose -s libgazebo_ros_init.so -s libgazebo_ros_factory.so \
  book/examples/humanoid-sim/gazebo_models/simple_world.sdf &
```

### Spawn the Robot

In a new terminal:

```bash
source /opt/ros/humble/setup.bash

# Spawn the humanoid robot
ros2 service call /spawn_entity gazebo_msgs/SpawnEntity \
  "{name: 'humanoid_simple', xml: '$(cat book/examples/humanoid-sim/gazebo_models/humanoid_simple.urdf)'}"
```

### Run Control Nodes

```bash
# Terminal 3: Run the humanoid controller
cd ~/PhysicalAI-Book
python3 book/static/code-examples/ros2_humanoid_nodes.py
```

### Monitor Topics

```bash
# Terminal 4: Monitor joint states
ros2 topic hz /joint_states

# Terminal 5: Monitor joint commands
ros2 topic echo /joint_commands --once
```

## Troubleshooting

### Issue 1: Gazebo GUI Doesn't Appear

**Symptoms**: Gazebo starts but no window opens

**Solutions**:
```bash
# Check if X11 is forwarded correctly
echo $DISPLAY
# Should output: :0 or :0.0

# If empty, add to ~/.bashrc:
export DISPLAY=:0

# Test with a simple X11 app
xeyes  # Should open a window with eyes that follow your cursor

# If still no luck, check GPU acceleration
glxinfo | grep "direct rendering"
# If "No", GPU acceleration is disabled (slower but OK)
```

### Issue 2: ROS 2 Bridge Not Found

**Symptoms**: `gazebo: symbol lookup error: ... gazebo_ros`

**Solutions**:
```bash
# Verify gazebo_ros is installed
ros2 pkg find gazebo_ros

# If not found, reinstall:
sudo apt-get install ros-humble-gazebo-ros

# Update environment:
source /opt/ros/humble/setup.bash
```

### Issue 3: URDF Syntax Errors

**Symptoms**: `Error parsing URDF, failed to load model`

**Solutions**:
```bash
# Validate URDF
check_urdf your_robot.urdf

# Common errors:
# - Duplicate joint names → rename
# - Negative inertia → check math (Ixx = m*(d²+h²)/12)
# - Circular dependencies → draw kinematic chain

# Use XML linter:
xmllint --noout your_robot.urdf
```

### Issue 4: Slow Physics Simulation

**Symptoms**: Real-time factor < 1.0 (Gazebo slower than real-time)

**Solutions**:
```bash
# In the world file, optimize physics:
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <!-- Try 0.002 for faster but less accurate -->

# Reduce sensor update rates:
<sensor type="camera" name="camera">
  <update_rate>10</update_rate>  <!-- Not 30 -->
</sensor>

# Use fewer objects in the world
# Use GPU-accelerated LiDAR (gpu_lidar, not cpu_lidar)
```

### Issue 5: WSL2 Audio/GUI Issues

**Symptoms**: Gazebo runs but crashes or GUI is choppy

**Solutions**:
```bash
# Disable audio (not needed for robotics)
export ALSA_CARD=dummy

# Use software rendering if GPU issues:
export LIBGL_ALWAYS_INDIRECT=1
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins

# Increase WSL2 resources in Windows PowerShell:
# Create/edit ~/.wslconfig:
[wsl2]
memory=8GB
processors=4
swap=2GB
```

## Development Workflow

### Quick Start Commands

```bash
# Setup (run once)
cd ~/PhysicalAI-Book
source /opt/ros/humble/setup.bash

# Terminal 1: Start Gazebo
gazebo --verbose -s libgazebo_ros_init.so -s libgazebo_ros_factory.so \
  book/examples/humanoid-sim/gazebo_models/simple_world.sdf &

# Terminal 2: Spawn robot
sleep 2  # Wait for Gazebo to start
ros2 service call /spawn_entity gazebo_msgs/SpawnEntity \
  "{name: 'humanoid', xml: '$(cat book/examples/humanoid-sim/gazebo_models/humanoid_simple.urdf)'}"

# Terminal 3: Run controller
python3 book/static/code-examples/ros2_humanoid_nodes.py

# Terminal 4: Monitor (optional)
ros2 topic hz /joint_states
```

### Running Tests

```bash
# Run all Module 2 tests
cd ~/PhysicalAI-Book
pytest tests/unit/test_module2_examples.py -v

# Run specific test class
pytest tests/unit/test_module2_examples.py::TestModule2ExampleSyntax -v

# With coverage
pytest tests/unit/test_module2_examples.py --cov=book/static/code-examples
```

### Debugging Tips

**Enable verbose logging:**
```bash
# ROS 2 debug logging
export ROS_LOG_DIR=/tmp/ros_logs
ros2 run --prefix 'gdb -ex run -ex "handle SIG33 nostop noprint pass" --args' \
  humanoid_sim humanoid_controller
```

**Inspect Gazebo state:**
```bash
# Print all models and their properties
gz model -l

# Get specific model info
gz model -m humanoid_simple -i

# Get joint state
gz joint -m humanoid_simple -j shoulder_pitch
```

**Visualize in RViz:**
```bash
rviz2 -d ~/PhysicalAI-Book/rviz_config.rviz
```

## Platform-Specific Notes

### Ubuntu 20.04
- Gazebo 11 is the standard
- ROS 2 Foxy (older) or use ROS 2 Humble from source
- Most stable for production

### Ubuntu 22.04
- Gazebo 11 still supported
- Native ROS 2 Humble support
- Recommended for new projects

### macOS
- Gazebo has limited macOS support
- Consider Docker or Linux VM
- Alternative: MuJoCo (easier on macOS)

### Windows (WSL2)
- Works but requires X11 server
- Performance ~80% of native Linux
- Good for development/testing

## Next Steps

1. **Run the exercises** in Module 2 to practice with Gazebo
2. **Modify the URDF** to understand robot modeling
3. **Inspect sensor output** to understand sim-to-real transfer
4. **Proceed to Module 3** for perception and Isaac Sim

## Resources

- **Official Gazebo Docs**: http://gazebosim.org/
- **ROS 2 + Gazebo Integration**: https://github.com/ros-simulation/gazebo_ros_pkgs
- **URDF Tutorial**: http://wiki.ros.org/urdf
- **Gazebo Community Forum**: https://community.osrfoundation.org/c/simulation/gazebo/
- **ROS 2 Discourse**: https://discourse.ros.org/

## Getting Help

If you encounter issues:

1. **Check error messages** — Gazebo prints detailed error messages
2. **Search the community** — Many issues have known solutions
3. **Verify installation** — Re-run installation steps
4. **Check dependencies** — Ensure ROS 2, Gazebo, and plugins are compatible
5. **Post on ROS Discourse** — Include error logs, ROS/Gazebo versions, OS

---

**Ready to run the exercises?** → [Module 2 Exercises](exercises.md)

You're all set! Gazebo is now configured and ready for humanoid robotics simulation.
