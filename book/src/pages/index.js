import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs">
            Start Learning
          </Link>
        </div>
      </div>
    </header>
  );
}

const FeatureList = [
  {
    title: 'ROS 2: The Robotic Nervous System',
    icon: '🤖',
    description: (
      <>
        Master ROS 2 nodes, topics, services, and actions. Bridge Python AI
        agents to robot controllers using rclpy and URDF.
      </>
    ),
  },
  {
    title: 'Digital Twins with Gazebo & Unity',
    icon: '🎮',
    description: (
      <>
        Build physics-accurate simulations in Gazebo and photorealistic
        environments in Unity for human-robot interaction.
      </>
    ),
  },
  {
    title: 'NVIDIA Isaac & Navigation',
    icon: '🧭',
    description: (
      <>
        Deploy GPU-accelerated perception with Isaac ROS, VSLAM, and Nav2
        for autonomous bipedal locomotion and balance control.
      </>
    ),
  },
  {
    title: 'Voice-to-Action with VLA',
    icon: '🗣️',
    description: (
      <>
        Integrate Whisper for speech recognition and GPT for cognitive
        planning to create conversational humanoid robots.
      </>
    ),
  },
];

function Feature({icon, title, description}) {
  return (
    <div className={clsx('col col--3')}>
      <div className={styles.featureCard}>
        <div className={styles.featureIcon}>{icon}</div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

const ModuleList = [
  {
    title: 'Module 1: The Robotic Nervous System',
    link: '/docs/module-1-ros2/intro',
    description: 'ROS 2 architecture, nodes, topics, services, Python agents, and URDF for humanoids.',
  },
  {
    title: 'Module 2: The Digital Twin',
    link: '/docs/module-2-digital-twin/intro',
    description: 'Gazebo physics simulation, Unity integration, and sensor simulation.',
  },
  {
    title: 'Module 3: The AI-Robot Brain',
    link: '/docs/module-3-isaac/intro',
    description: 'NVIDIA Isaac, VSLAM, Nav2 for bipedal robots, and humanoid kinematics.',
  },
  {
    title: 'Module 4: Vision-Language-Action',
    link: '/docs/module-4-vla/intro',
    description: 'Whisper voice-to-action, GPT planning, conversational robotics, and capstone project.',
  },
];

function ModuleCard({title, link, description}) {
  return (
    <div className={clsx('col col--6')}>
      <Link to={link} className={styles.moduleLink}>
        <div className={styles.moduleCard}>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
      </Link>
    </div>
  );
}

function HomepageModules() {
  return (
    <section className={styles.modules}>
      <div className="container">
        <h2 className={styles.sectionTitle}>Course Modules</h2>
        <div className="row">
          {ModuleList.map((props, idx) => (
            <ModuleCard key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="A Comprehensive Technical Textbook on Embodied Intelligence, ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action Systems">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <HomepageModules />
      </main>
    </Layout>
  );
}
