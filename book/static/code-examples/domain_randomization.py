"""
Domain Randomization for Sim-to-Real Transfer.

This module demonstrates domain randomization techniques for robotic grasping.
Randomizes world parameters during training to build robustness for real hardware.

Learning Goals:
  - Understand how to implement domain randomization
  - See parameter variation ranges for realistic sim-to-real transfer
  - Learn how to apply randomization in training loops
  - Practice debugging with domain randomization

Example:
  >>> randomizer = DomainRandomizer()
  >>> for episode in range(1000):
  ...     params = randomizer.sample()
  ...     world = randomizer.apply_to_gazebo(params)
  ...     train_episode(world, params)
"""

import numpy as np
from typing import Dict, List, Tuple
import json


class DomainRandomizer:
    """Generates randomized world parameters for training."""

    def __init__(self, randomization_config: str = None):
        """
        Initialize domain randomizer with configuration.

        Args:
            randomization_config: Path to JSON config with parameter ranges
                                If None, uses conservative defaults
        """
        if randomization_config:
            with open(randomization_config, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()

    @staticmethod
    def _default_config() -> Dict:
        """
        Default randomization configuration.

        These ranges match real-world variation observed in robotic grasping.
        Customize based on your specific robot and task.
        """
        return {
            # Object properties (what varies in real world)
            'object_friction': {
                'type': 'uniform',
                'min': 0.3,  # Glass (slippery)
                'max': 1.0,  # Rubber (grippy)
                'unit': 'coefficient (dimensionless)',
                'real_world_note': 'Different materials, dust, wet surfaces'
            },
            'object_mass': {
                'type': 'normal',
                'mean': 1.0,
                'std': 0.1,  # ±10% manufacturing variation
                'unit': 'kg',
                'real_world_note': 'Manufacturing tolerance, fill level variation'
            },
            'object_position_x': {
                'type': 'normal',
                'mean': 0.5,
                'std': 0.05,  # ±5cm (±1 std)
                'unit': 'm',
                'real_world_note': 'Placement precision'
            },
            'object_position_y': {
                'type': 'normal',
                'mean': 0.1,
                'std': 0.05,
                'unit': 'm',
                'real_world_note': 'Centering accuracy'
            },

            # Sensor properties
            'camera_noise_std': {
                'type': 'uniform',
                'min': 0.0,
                'max': 0.05,  # ±5% pixel noise
                'unit': 'fraction of pixel value [0-1]',
                'real_world_note': 'Sensor noise varies with ISO gain, lighting'
            },
            'camera_latency_ms': {
                'type': 'normal',
                'mean': 33,  # Typical camera frame time
                'std': 5,    # ±5ms jitter
                'unit': 'milliseconds',
                'real_world_note': 'Rolling shutter, processing delay'
            },

            # Lighting
            'lighting_intensity': {
                'type': 'uniform',
                'min': 0.5,  # Dim (overcast/evening)
                'max': 1.5,  # Bright (sunny/fluorescent)
                'unit': 'relative intensity',
                'real_world_note': 'Time of day, weather, indoor/outdoor'
            },
            'lighting_direction_x': {
                'type': 'uniform',
                'min': -1.0,
                'max': 1.0,
                'unit': 'normalized direction component',
                'real_world_note': 'Sun angle varies with latitude/season'
            },
            'lighting_direction_y': {
                'type': 'uniform',
                'min': -1.0,
                'max': 1.0,
                'unit': 'normalized direction component',
                'real_world_note': 'Sun angle varies with time of day'
            },

            # Motor/actuator properties
            'motor_delay_ms': {
                'type': 'uniform',
                'min': 0,
                'max': 50,  # 0-50ms motor latency
                'unit': 'milliseconds',
                'real_world_note': 'Servo response time, communication lag'
            },
            'motor_max_velocity_factor': {
                'type': 'uniform',
                'min': 0.8,  # 80% of nominal
                'max': 1.2,  # 120% of nominal
                'unit': 'multiplication factor',
                'real_world_note': 'Motor wear, temperature effects, voltage variation'
            },

            # Gripper properties
            'gripper_friction': {
                'type': 'uniform',
                'min': 0.3,  # Worn/smooth fingers
                'max': 0.9,  # Fresh/rough fingers
                'unit': 'coefficient',
                'real_world_note': 'Gripper finger wear, contamination'
            },

            # Environment
            'table_friction': {
                'type': 'uniform',
                'min': 0.3,  # Wet, dusty, worn
                'max': 0.9,  # Clean, new
                'unit': 'coefficient',
                'real_world_note': 'Table surface condition'
            },

            # Randomization strategy flags
            'randomization_enabled': True,
            'log_samples': False,  # Set to True for debugging
        }

    def sample(self) -> Dict[str, float]:
        """
        Sample a new random world configuration.

        Returns:
            Dictionary mapping parameter names to randomized values
        """
        params = {}

        for param_name, param_spec in self.config.items():
            # Skip metadata fields
            if param_name in ['randomization_enabled', 'log_samples']:
                continue

            if not self.config['randomization_enabled']:
                # Return nominal values
                if param_spec['type'] == 'uniform':
                    params[param_name] = param_spec['min']
                elif param_spec['type'] == 'normal':
                    params[param_name] = param_spec['mean']
            else:
                # Sample random value
                if param_spec['type'] == 'uniform':
                    params[param_name] = np.random.uniform(
                        param_spec['min'],
                        param_spec['max']
                    )
                elif param_spec['type'] == 'normal':
                    params[param_name] = np.random.normal(
                        param_spec['mean'],
                        param_spec['std']
                    )

        # Normalize lighting direction (must be unit vector)
        light_x = params.get('lighting_direction_x', 0)
        light_y = params.get('lighting_direction_y', 0)
        light_z = 1.0  # Always down (standard)

        light_norm = np.sqrt(light_x**2 + light_y**2 + light_z**2)
        params['lighting_direction'] = [
            light_x / light_norm,
            light_y / light_norm,
            light_z / light_norm
        ]

        if self.config['log_samples']:
            print(f"Sampled randomization: {json.dumps(params, indent=2)}")

        return params

    def apply_to_gazebo(self, params: Dict[str, float]) -> str:
        """
        Generate Gazebo SDF world file with randomized parameters.

        Args:
            params: Parameter dictionary from sample()

        Returns:
            SDF world content as string (ready to write to file)
        """
        sdf_template = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="grasping_world">
    <!-- Physics engine -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <pose>
        {lighting_x}
        {lighting_y}
        {lighting_z}
        0 0 0
      </pose>
      <diffuse>{intensity} {intensity} {intensity} 1</diffuse>
      <specular>{intensity_spec} {intensity_spec} {intensity_spec} 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>
        {light_dir_x}
        {light_dir_y}
        {light_dir_z}
      </direction>
      <cast_shadows>1</cast_shadows>
    </light>

    <light name="ambient" type="ambient">
      <ambient>{ambient} {ambient} {ambient} 1</ambient>
    </light>

    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>{table_friction}</mu>
                <mu2>{table_friction}</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Grey</name>
            </script>
          </material>
        </visual>
      </link>
    </model>

    <!-- Table -->
    <model name="table">
      <pose>0.5 0.1 0.4 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.8 0.8 0.8</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>{table_friction}</mu>
                <mu2>{table_friction}</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.8 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Wood</name>
            </script>
          </material>
        </visual>
      </link>
    </model>

    <!-- Grasping Object (Bottle) - RANDOMIZED -->
    <model name="bottle">
      <pose>
        {obj_x}
        {obj_y}
        0.9
        0 0 0
      </pose>
      <link name="link">
        <inertial>
          <mass>{obj_mass}</mass>
          <inertia>
            <ixx>{ixx}</ixx>
            <iyy>{iyy}</iyy>
            <izz>{izz}</izz>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyz>0</iyz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.04</radius>
              <length>0.2</length>
            </cylinder>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>{object_friction}</mu>
                <mu2>{object_friction}</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.04</radius>
              <length>0.2</length>
            </cylinder>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Red</name>
            </script>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
"""

        # Calculate inertia for cylinder: I = (1/12) * m * (3*r^2 + h^2)
        m = params['object_mass']
        r = 0.04  # radius
        h = 0.2   # height
        ixx = (1/12) * m * (3*r**2 + h**2)
        iyy = ixx
        izz = 0.5 * m * r**2

        # Normalize lighting direction
        light_dir = params['lighting_direction']
        intensity = params['lighting_intensity']

        # Render template
        sdf = sdf_template.format(
            # Object
            obj_x=params['object_position_x'],
            obj_y=params['object_position_y'],
            obj_mass=params['object_mass'],
            ixx=ixx,
            iyy=iyy,
            izz=izz,
            object_friction=params['object_friction'],
            # Lighting
            lighting_x=light_dir[0],
            lighting_y=light_dir[1],
            lighting_z=light_dir[2],
            light_dir_x=light_dir[0],
            light_dir_y=light_dir[1],
            light_dir_z=light_dir[2],
            intensity=intensity,
            intensity_spec=intensity * 0.8,
            ambient=intensity * 0.3,
            # Table
            table_friction=params['table_friction'],
        )

        return sdf

    def get_statistics(self, num_samples: int = 10000) -> Dict:
        """
        Analyze randomization statistics (useful for validation).

        Args:
            num_samples: Number of samples to analyze

        Returns:
            Statistics dictionary with mean, std, min, max for each parameter
        """
        samples = [self.sample() for _ in range(num_samples)]

        stats = {}
        for key in samples[0].keys():
            values = [s[key] for s in samples]
            # Skip vector parameters (lighting_direction)
            if isinstance(values[0], (int, float)):
                stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }

        return stats


def main():
    """Demonstrate domain randomization usage."""
    print("=" * 70)
    print("Domain Randomization for Robotic Grasping")
    print("=" * 70)

    # Create randomizer
    randomizer = DomainRandomizer()

    # Sample and display randomization
    print("\n1. Sample Randomized Parameters:")
    print("-" * 70)
    params = randomizer.sample()
    for key, value in sorted(params.items()):
        if isinstance(value, (int, float)):
            print(f"  {key:30s}: {value:.4f}")

    # Generate SDF world
    print("\n2. Generate Gazebo World (SDF):")
    print("-" * 70)
    sdf = randomizer.apply_to_gazebo(params)
    print("Generated SDF world (first 500 chars):")
    print(sdf[:500] + "...")

    # Save to file
    with open('/tmp/randomized_world.sdf', 'w') as f:
        f.write(sdf)
    print(f"\nSaved to /tmp/randomized_world.sdf")

    # Analyze statistics
    print("\n3. Randomization Statistics (10,000 samples):")
    print("-" * 70)
    stats = randomizer.get_statistics(num_samples=10000)
    for param, stat_dict in sorted(stats.items()):
        print(f"\n  {param}:")
        print(f"    Mean: {stat_dict['mean']:.4f}")
        print(f"    Std:  {stat_dict['std']:.4f}")
        print(f"    Min:  {stat_dict['min']:.4f}")
        print(f"    Max:  {stat_dict['max']:.4f}")

    print("\n" + "=" * 70)
    print("Usage in Training Loop:")
    print("=" * 70)
    print("""
for episode in range(1000):
    # Sample new world
    params = randomizer.sample()

    # Apply to Gazebo
    sdf = randomizer.apply_to_gazebo(params)
    with open('world.sdf', 'w') as f:
        f.write(sdf)

    # Launch Gazebo with randomized world
    # gazebo world.sdf &

    # Train policy in this episode
    # ...
    """)


if __name__ == '__main__':
    main()
