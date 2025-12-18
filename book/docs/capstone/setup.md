# Capstone Project: Setup Guide

This guide provides step-by-step instructions for setting up the environment to run the full capstone project. The project is designed to run in a ROS 2 and Gazebo environment on a Linux-based system (or WSL 2 on Windows).

## Prerequisites

Before you begin, ensure you have the following installed:

*   **ROS 2 Humble Hawksbill or Jazzy Jalisco**: Follow the official [ROS 2 installation guide](https://docs.ros.org/en/humble/Installation.html). Make sure to install `ros-dev-tools`.
*   **Gazebo**: Gazebo is typically installed as part of the "desktop" ROS 2 installation. Verify it's installed by running `gazebo`.
*   **Git**: For cloning the project repository.
*   **Python 3.8+**: With `pip` and `venv`.

## 1. Clone the Repository

First, clone the project repository to your local machine.

```bash
git clone https://github.com/your-username/PhysicalAI-Book.git
cd PhysicalAI-Book
```

## 2. Set up the ROS 2 Workspace

The capstone project is structured as a ROS 2 workspace.

```bash
# Create a new ROS 2 workspace
mkdir -p ros2_ws/src
cd ros2_ws

# Create a symbolic link to the capstone project examples
ln -s ../book/examples/humanoid-sim src/humanoid-sim

# You would also link any custom interface packages here
# ln -s ../interfaces src/interfaces
```

## 3. Install Dependencies

### Python Dependencies

The Python packages required for the project are listed in `requirements.txt`.

```bash
# Navigate to the examples directory
cd ../book/examples

# Install Python requirements
pip install -r requirements.txt
```

This will install libraries like `speech_recognition`, `numpy`, etc.

### ROS 2 Dependencies

Use `rosdep` to install any missing ROS 2 system dependencies.

```bash
# Navigate to the root of your ROS 2 workspace
cd ../ros2_ws

# Initialize rosdep (if you haven't already)
sudo rosdep init
rosdep update

# Install dependencies for the packages in your workspace
rosdep install -i --from-path src --rosdistro humble -y
```

## 4. Build the Workspace

Once all dependencies are installed, you can build the ROS 2 workspace using `colcon`.

```bash
# From the root of your ros2_ws
colcon build
```

If the build is successful, you will see `install`, `build`, and `log` directories in your workspace.

## 5. Source the Workspace

Before you can run any of the nodes, you need to source the workspace's setup file. This adds the compiled packages to your ROS 2 environment.

```bash
# From the root of your ros2_ws
source install/setup.bash
```

**Tip**: Add this command to your `~/.bashrc` file to automatically source the workspace in new terminal sessions.

## (Optional) Docker Devcontainer for Reproducibility

For a fully reproducible environment, a Docker devcontainer is provided. If you are using VS Code with the "Dev Containers" extension, you can simply open the project folder and VS Code will ask if you want to "Reopen in Container".

This will build a Docker image with ROS 2, Gazebo, and all dependencies pre-installed, providing a consistent environment for everyone.

## Troubleshooting Common Issues

*   **`colcon build` fails**:
    *   Make sure you have sourced your main ROS 2 installation (`source /opt/ros/humble/setup.bash`).
    *   Run `rosdep install` again to ensure all system dependencies are present.
*   **`pip install` fails**:
    *   Ensure you are using a compatible Python version.
    *   Some packages may have system-level dependencies (like `portaudio` for `pyaudio`). Check the error messages for clues.
*   **Gazebo doesn't launch**:
    *   Ensure your graphics drivers are installed and configured correctly.
    *   If in WSL, make sure you have set up GUI forwarding.