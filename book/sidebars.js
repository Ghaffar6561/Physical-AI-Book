/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a set of docs in the sidebar
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // But you can create a sidebar manually
  tutorialSidebar: [
    'intro',
    {
	type: 'category',
      label: 'Module 1: Physical AI Foundations',
      collapsible: true,
      collapsed: false,
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
      collapsible: true,
      collapsed: false,
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
      collapsible: true,
      collapsed: false,
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
      collapsible: true,
      collapsed: false,
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
      label: 'Module 5: Capstone Project',
      collapsible: true,
      collapsed: false,
      items: [
        '05-capstone/architecture',
        '05-capstone/setup',
        '05-capstone/running-the-system',
        '05-capstone/extensions',
      ],
    },
    {
	type: 'category',
      label: 'Appendices',
      collapsible: true,
      collapsed: true,
      items: [
        'glossary',
        'references',
        'troubleshooting',
      ],
    },
  ],
};

module.exports = sidebars;
