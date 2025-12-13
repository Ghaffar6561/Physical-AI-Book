// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A Comprehensive Technical Textbook on Embodied Intelligence',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://asad.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/PhysicalAI-Book/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'asad', // Usually your GitHub org/username.
  projectName: 'PhysicalAI-Book', // Usually your repo name.
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is in Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl: 'https://github.com/asad/PhysicalAI-Book/tree/001-physical-ai-book/book/',
          path: 'docs',
          routeBasePath: '/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/social-card.png',
      metadata: [
        {
          name: 'description',
          content: 'Learn Physical AI and humanoid robotics through comprehensive technical modules covering ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems.',
        },
      ],
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Modules',
          },
          {
            href: 'https://github.com/asad/PhysicalAI-Book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Modules',
            items: [
              {
                label: 'Foundations',
                to: '/01-foundations/intro',
              },
              {
                label: 'Simulation',
                to: '/02-simulation/intro',
              },
              {
                label: 'Perception',
                to: '/03-perception/intro',
              },
              {
                label: 'VLA Systems',
                to: '/04-vla-systems/intro',
              },
              {
                label: 'Capstone',
                to: '/05-capstone/architecture',
              },
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'ROS 2 Documentation',
                href: 'https://docs.ros.org/en/humble/',
              },
              {
                label: 'Gazebo Docs',
                href: 'https://gazebosim.org/',
              },
              {
                label: 'NVIDIA Isaac',
                href: 'https://developer.nvidia.com/isaac/',
              },
              {
                label: 'GitHub Repository',
                href: 'https://github.com/asad/PhysicalAI-Book',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'ROS Discourse',
                href: 'https://discourse.ros.org/',
              },
              {
                label: 'Gazebo Forum',
                href: 'https://community.gazebosim.org/',
              },
              {
                label: 'Issues & Feedback',
                href: 'https://github.com/asad/PhysicalAI-Book/issues',
              },
            ],
          },
        ],
        copyright: `Copyright © 2025 Physical AI Textbook Project. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'yaml', 'bash', 'cpp', 'java', 'xml'],
      },
    }),
};

module.exports = config;
