import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/docs',
    component: ComponentCreator('/docs', '9fe'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '9d2'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', 'b13'),
            routes: [
              {
                path: '/docs/',
                component: ComponentCreator('/docs/', '0fd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/glossary',
                component: ComponentCreator('/docs/glossary', 'daa'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/exercises',
                component: ComponentCreator('/docs/module-1-ros2/exercises', 'c4d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/intro',
                component: ComponentCreator('/docs/module-1-ros2/intro', 'ff7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/launch-files',
                component: ComponentCreator('/docs/module-1-ros2/launch-files', '677'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/nodes-topics-services',
                component: ComponentCreator('/docs/module-1-ros2/nodes-topics-services', '87b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/physical-ai-foundations',
                component: ComponentCreator('/docs/module-1-ros2/physical-ai-foundations', 'e35'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/rclpy-python-agents',
                component: ComponentCreator('/docs/module-1-ros2/rclpy-python-agents', '508'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/ros2-architecture',
                component: ComponentCreator('/docs/module-1-ros2/ros2-architecture', '0de'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-1-ros2/urdf-humanoids',
                component: ComponentCreator('/docs/module-1-ros2/urdf-humanoids', 'f03'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/exercises',
                component: ComponentCreator('/docs/module-2-digital-twin/exercises', '3fd'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/gazebo-fundamentals',
                component: ComponentCreator('/docs/module-2-digital-twin/gazebo-fundamentals', '110'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/intro',
                component: ComponentCreator('/docs/module-2-digital-twin/intro', '945'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/setup-gazebo',
                component: ComponentCreator('/docs/module-2-digital-twin/setup-gazebo', 'bb3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-2-digital-twin/unity-integration',
                component: ComponentCreator('/docs/module-2-digital-twin/unity-integration', '083'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-isaac/bipedal-locomotion',
                component: ComponentCreator('/docs/module-3-isaac/bipedal-locomotion', '5bf'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-isaac/exercises',
                component: ComponentCreator('/docs/module-3-isaac/exercises', '354'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-isaac/humanoid-kinematics',
                component: ComponentCreator('/docs/module-3-isaac/humanoid-kinematics', '353'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-isaac/intro',
                component: ComponentCreator('/docs/module-3-isaac/intro', '449'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-isaac/isaac-ros-vslam',
                component: ComponentCreator('/docs/module-3-isaac/isaac-ros-vslam', 'd75'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-isaac/isaac-sim',
                component: ComponentCreator('/docs/module-3-isaac/isaac-sim', 'a03'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-isaac/nav2-bipedal',
                component: ComponentCreator('/docs/module-3-isaac/nav2-bipedal', '9aa'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-isaac/sensor-fusion',
                component: ComponentCreator('/docs/module-3-isaac/sensor-fusion', '86b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-isaac/setup-isaac',
                component: ComponentCreator('/docs/module-3-isaac/setup-isaac', 'cbb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-3-isaac/sim-to-real-transfer',
                component: ComponentCreator('/docs/module-3-isaac/sim-to-real-transfer', 'bef'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/action-grounding',
                component: ComponentCreator('/docs/module-4-vla/action-grounding', 'c96'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/capstone-project',
                component: ComponentCreator('/docs/module-4-vla/capstone-project', 'd54'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/conversational-robotics',
                component: ComponentCreator('/docs/module-4-vla/conversational-robotics', '1d9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/embodied-reasoning',
                component: ComponentCreator('/docs/module-4-vla/embodied-reasoning', '88a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/end-to-end-pipeline',
                component: ComponentCreator('/docs/module-4-vla/end-to-end-pipeline', 'fd4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/exercises',
                component: ComponentCreator('/docs/module-4-vla/exercises', 'e95'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/intro',
                component: ComponentCreator('/docs/module-4-vla/intro', '12c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/llm-planning',
                component: ComponentCreator('/docs/module-4-vla/llm-planning', 'eba'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/lora-adaptation',
                component: ComponentCreator('/docs/module-4-vla/lora-adaptation', '81e'),
                exact: true
              },
              {
                path: '/docs/module-4-vla/prompting-strategies',
                component: ComponentCreator('/docs/module-4-vla/prompting-strategies', '372'),
                exact: true
              },
              {
                path: '/docs/module-4-vla/setup-llm',
                component: ComponentCreator('/docs/module-4-vla/setup-llm', 'f6e'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/vision-language-models',
                component: ComponentCreator('/docs/module-4-vla/vision-language-models', '90d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/vla-architecture',
                component: ComponentCreator('/docs/module-4-vla/vla-architecture', '14b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module-4-vla/voice-to-action',
                component: ComponentCreator('/docs/module-4-vla/voice-to-action', '1cd'),
                exact: true
              },
              {
                path: '/docs/module-4-vla/whisper-integration',
                component: ComponentCreator('/docs/module-4-vla/whisper-integration', 'd15'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/references',
                component: ComponentCreator('/docs/references', '4a2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/troubleshooting',
                component: ComponentCreator('/docs/troubleshooting', '2bf'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
