const sidebars = {
  tutorialSidebar: [
    'intro',

    {
      type: 'category',
      label: 'Module 1: Physical AI Foundations',
      items: [
        'foundations/intro',
        'foundations/embodied-intelligence',
        'foundations/ros2-intro',
        'foundations/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 2: Digital Twins & Gazebo',
      items: [
        'simulation/intro',
        'simulation/gazebo-fundamentals',
        'simulation/urdf-humanoid',
        'simulation/setup-gazebo',
        'simulation/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 3: Perception & NVIDIA Isaac',
      items: [
        'perception/intro',
        'perception/sensor-fusion',
        'perception/sim-to-real-transfer',
        'perception/isaac-workflows',
        'perception/setup-isaac',
        'perception/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action Systems',
      items: [
        'vla-systems/intro',
        'vla-systems/llm-planning',
        'vla-systems/voice-to-action',
        'vla-systems/lora-adaptation',
        'vla-systems/setup-llm',
        'vla-systems/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 5: Capstone Project',
      items: [
        'capstone/architecture',
        'capstone/setup',
        'capstone/running-the-system',
        'capstone/extensions',
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
