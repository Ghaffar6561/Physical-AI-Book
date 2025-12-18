# Troubleshooting

This guide covers common issues and solutions for the Physical AI & Humanoid Robotics textbook.

## Setup Issues

### ROS 2 Installation
**Problem**: ROS 2 Humble/Jazzy not found
**Solution**: 
1. Verify installation: `source /opt/ros/humble/setup.bash` (or `jazzy`)
2. Add to your `.bashrc` or `.zshrc`: `source /opt/ros/humble/setup.bash`
3. Install from official guide: https://docs.ros.org/en/humble/Installation.html

### Gazebo Installation
**Problem**: Gazebo fails to start
**Solution**:
1. Check version: `gazebo --version` (should be 11+)
2. If using Ubuntu/WSL2, ensure X11 forwarding is set up
3. Install with: `sudo apt install gazebo libgazebo-dev`

### Python Dependencies
**Problem**: Python packages not installed
**Solution**:
1. Navigate to examples directory: `cd book/examples`
2. Install requirements: `pip install -r requirements.txt`
3. Or create virtual environment: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

## Docusaurus Issues

### Build Failures
**Problem**: `npm run build` fails with errors
**Solution**:
1. Clear node_modules: `rm -rf node_modules package-lock.json`
2. Reinstall: `npm install`
3. Check Node.js version (must be 16+)

### Local Server Fails
**Problem**: `npm start` fails or page doesn't load
**Solution**:
1. Check Node.js version: `node --version` (must be 16+)
2. Ensure no other process is using port 3000
3. Look for syntax errors in Markdown files

## ROS 2 Issues

### Node Communication
**Problem**: Nodes not communicating via topics/services
**Solution**:
1. Check that nodes are in the same ROS domain: `echo $ROS_DOMAIN_ID`
2. Verify topics exist: `ros2 topic list`
3. Check if nodes are running: `ros2 run <pkg_name> <node_name>`

### TF Transform Issues
**Problem**: Coordinate transforms not working
**Solution**:
1. Check TF tree: `ros2 run tf2_tools view_frames`
2. Verify transform publishers are running
3. Check frame names match between nodes

## Gazebo Simulation Issues

### Physics Instability
**Problem**: Robot falls through floor or exhibits unrealistic bouncing
**Solution**:
1. Reduce timestep in world file: `<max_step_size>0.0005</max_step_size>`
2. Verify inertia values in URDF (should be positive and realistic)
3. Check collision geometry matches visual geometry

### Sensors Producing No Data
**Problem**: Camera/LiDAR topics exist but no messages arriving
**Solution**:
1. Verify Gazebo plugins are loaded: `gzserver` with ROS plugins
2. Check plugin configuration in SDF/URDF
3. Verify sensor update rates are reasonable

### Slow Simulation Speed
**Problem**: Simulation runs in slow motion
**Solution**:
1. Check real_time_factor in world file: should be 1.0 or higher
2. Simplify collision geometry if too complex
3. Reduce sensor update rates in SDF

## Code Examples

### Python Import Errors
**Problem**: Import errors when running code examples
**Solution**:
1. Ensure `rclpy` is installed: `pip install rclpy`
2. Source ROS 2: `source /opt/ros/humble/setup.bash`
3. Check Python version: must be 3.8+

### Robot Commands Not Executing
**Problem**: Joint commands sent but robot doesn't move
**Solution**:
1. Verify joint names match between command and robot model
2. Check joint limits in URDF
3. Confirm control pipeline is running

## Vision-Language-Action (VLA) Issues

### LLM API Errors
**Problem**: LLM calls failing
**Solution**:
1. Check API key if using commercial service
2. Verify local LLM is running (Ollama): `ollama list`
3. Check network connectivity to API endpoint

### Speech Recognition Issues
**Problem**: Voice commands not recognized
**Solution**:
1. Check microphone permissions
2. Verify speech recognition library installed: `pip install speech-recognition`
3. Test with offline recognition if online models failing

## Module-Specific Troubleshooting

### Module 2 (Gazebo/URDF)
**Problem**: URDF fails to load in Gazebo
**Solution**:
1. Validate URDF syntax: `check_urdf path/to/robot.urdf`
2. Check for joint loops that create circular dependencies
3. Verify all mesh files exist and path is correct

### Module 3 (Sim-to-Real Transfer)
**Problem**: Domain randomization parameters not affecting simulation
**Solution**:
1. Verify randomization code is executed at episode start
2. Check that randomization parameters are passed to Gazebo
3. Confirm randomization ranges are reasonable

### Module 4 (VLA Systems)
**Problem**: Language model produces nonsensical actions
**Solution**:
1. Verify prompt format matches model expectations
2. Check for hallucination with fact-checking step
3. Implement safety validation on generated actions

## Capstone Project Issues

### Nodes Not Starting
**Problem**: Capstone launch fails
**Solution**:
1. Check all dependencies are installed
2. Verify all ROS 2 packages in workspace are built
3. Source workspace: `source install/setup.bash`

### Performance Issues
**Problem**: Capstone system too slow for real-time operation
**Solution**:
1. Run LLM inference on GPU if available
2. Optimize perception pipeline for speed vs. accuracy
3. Implement multi-threading where appropriate

### Integration Failures
**Problem**: Individual modules work but capstone system fails
**Solution**:
1. Check data format compatibility between modules
2. Verify timing constraints (latency budgets)
3. Implement proper error handling for module failures

## Common Error Messages and Solutions

### "Package 'xyz' not found"
**Cause**: ROS 2 package not installed or not sourced
**Solution**: 
- Check if package exists: `find /opt/ros -name "*xyz*"`
- Install if available: `sudo apt install ros-humble-xyz`
- Source workspace if in development: `source install/setup.bash`

### "Address in use" or "Port already in use"
**Cause**: Service/program already running
**Solution**: Find process using port: `sudo lsof -i :port_num` and kill if needed

### "ImportError: No module named 'rclpy'"
**Cause**: ROS 2 Python packages not installed/available
**Solution**: 
1. Install ROS 2 Python packages: `sudo apt install python3-rosdep python3-rosinstall python3-vcstools`
2. Source ROS: `source /opt/ros/humble/setup.bash`
3. Check Python path: `python -c "import rclpy"`

## Getting Help

### Where to Ask Questions
1. Check existing issues on GitHub: [asad/PhysicalAI-Book](https://github.com/asad/PhysicalAI-Book)
2. Ask on ROS Answers: https://answers.ros.org/
3. For specific problems, create a new issue on GitHub with:
   - Version of software used
   - Exact error message
   - Steps to reproduce
   - What you've already tried

### Debugging Tips
1. Use `ros2 doctor` to diagnose system configuration
2. Check logs: `~/.ros/log/` or `~/.gazebo/log/`
3. Use `rqt_console` for ROS2 log messages
4. Enable debug output with `--ros-args --log-level debug`

## Performance Optimization

### For Slow Simulations
- Reduce physics update rate if not critical
- Simplify collision geometry (use boxes instead of complex meshes)
- Reduce sensor resolution temporarily for testing

### For High CPU/Memory Usage
- Close unnecessary applications
- Use lightweight VM/WSL2 configuration if applicable
- Monitor resource usage: `htop`

This troubleshooting guide will be updated as new issues are encountered. If you face a problem not covered here, please report it in the repository issues.