const sidebars = {
  tutorialSidebar: [
    'index',

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
      label: 'Module 5: End-to-End Learning & Diffusion Models',
      items: [
        'embodied-learning/intro',
        'embodied-learning/learning-spectrum',
        'embodied-learning/imitation-learning',
        'embodied-learning/reinforcement-learning',
        'embodied-learning/diffusion-for-robotics',
        'embodied-learning/end-to-end-learning',
        'embodied-learning/training-pipeline',
        'embodied-learning/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 6: Scaling Systems & Production',
      items: [
        'scaling-systems/intro',
        'scaling-systems/distributed-training',
        'scaling-systems/multi-task-learning',
        'scaling-systems/scaling-pipeline',
        'scaling-systems/benchmarking-framework',
        'scaling-systems/cost-analysis',
        'scaling-systems/fleet-architecture',
        'scaling-systems/real-robot-deployment',
        'scaling-systems/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 7: Capstone & Real-World Deployment',
      items: [
        'capstone-deployment/intro',
        'capstone-deployment/production-architecture',
        'capstone-deployment/deployment-strategies',
        'capstone-deployment/operations-maintenance',
        'capstone-deployment/case-studies',
      ],
    },

    {
      type: 'category',
      label: 'Capstone Project',
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
