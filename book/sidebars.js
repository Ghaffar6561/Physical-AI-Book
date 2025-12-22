const sidebars = {
  tutorialSidebar: [
    'index',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2/intro',
        'module-1-ros2/physical-ai-foundations',
        'module-1-ros2/ros2-architecture',
        'module-1-ros2/nodes-topics-services',
        'module-1-ros2/rclpy-python-agents',
        'module-1-ros2/launch-files',
        'module-1-ros2/urdf-humanoids',
        'module-1-ros2/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin/intro',
        'module-2-digital-twin/gazebo-fundamentals',
        'module-2-digital-twin/setup-gazebo',
        'module-2-digital-twin/unity-integration',
        'module-2-digital-twin/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-isaac/intro',
        'module-3-isaac/isaac-sim',
        'module-3-isaac/isaac-ros-vslam',
        'module-3-isaac/nav2-bipedal',
        'module-3-isaac/sensor-fusion',
        'module-3-isaac/sim-to-real-transfer',
        'module-3-isaac/humanoid-kinematics',
        'module-3-isaac/bipedal-locomotion',
        'module-3-isaac/setup-isaac',
        'module-3-isaac/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/intro',
        'module-4-vla/vision-language-models',
        'module-4-vla/vla-architecture',
        'module-4-vla/whisper-integration',
        'module-4-vla/llm-planning',
        'module-4-vla/conversational-robotics',
        'module-4-vla/action-grounding',
        'module-4-vla/embodied-reasoning',
        'module-4-vla/end-to-end-pipeline',
        'module-4-vla/capstone-project',
        'module-4-vla/setup-llm',
        'module-4-vla/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Appendices',
      items: [
        'glossary',
        'references',
        'troubleshooting',
      ],
    },
  ],
};

module.exports = sidebars;
