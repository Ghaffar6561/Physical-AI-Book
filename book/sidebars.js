const sidebars = {
  tutorialSidebar: [
    'intro',

    {
      type: 'category',
      label: 'Module 1: Physical AI Foundations',
      items: [
        '01-foundations/intro',
        '01-foundations/embodied-intelligence',
        '01-foundations/ros2-intro',
        '01-foundations/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 2: Digital Twins & Gazebo',
      items: [
        '02-simulation/intro',
        '02-simulation/gazebo-fundamentals',
        '02-simulation/urdf-humanoid',
        '02-simulation/setup-gazebo',
        '02-simulation/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 3: Perception & NVIDIA Isaac',
      items: [
        '03-perception/intro',
        '03-perception/sensor-fusion',
        '03-perception/sim-to-real-transfer',
        '03-perception/isaac-workflows',
        '03-perception/setup-isaac',
        '03-perception/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action Systems',
      items: [
        '04-vla-systems/intro',
        '04-vla-systems/llm-planning',
        '04-vla-systems/voice-to-action',
        '04-vla-systems/lora-adaptation',
        '04-vla-systems/setup-llm',
        '04-vla-systems/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 5: End-to-End Learning & Diffusion Models',
      items: [
        '05-embodied-learning/intro',
        '05-embodied-learning/learning-spectrum',
        '05-embodied-learning/imitation-learning',
        '05-embodied-learning/reinforcement-learning',
        '05-embodied-learning/diffusion-for-robotics',
        '05-embodied-learning/end-to-end-learning',
        '05-embodied-learning/training-pipeline',
        '05-embodied-learning/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 6: Scaling Systems & Production',
      items: [
        '06-scaling-systems/intro',
        '06-scaling-systems/distributed-training',
        '06-scaling-systems/multi-task-learning',
        '06-scaling-systems/scaling-pipeline',
        '06-scaling-systems/benchmarking-framework',
        '06-scaling-systems/cost-analysis',
        '06-scaling-systems/fleet-architecture',
        '06-scaling-systems/real-robot-deployment',
        '06-scaling-systems/exercises',
      ],
    },

    {
      type: 'category',
      label: 'Module 7: Capstone & Real-World Deployment',
      items: [
        '07-capstone-deployment/intro',
        '07-capstone-deployment/production-architecture',
        '07-capstone-deployment/deployment-strategies',
        '07-capstone-deployment/operations-maintenance',
        '07-capstone-deployment/case-studies',
      ],
    },

    {
      type: 'category',
      label: 'Capstone Project',
      items: [
        '08-capstone/architecture',
        '08-capstone/setup',
        '08-capstone/running-the-system',
        '08-capstone/extensions',
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
