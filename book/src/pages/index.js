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
    title: 'ROS 2 & Robotics Foundations',
    icon: '🤖',
    description: (
      <>
        Master the Robot Operating System 2, understand embodied intelligence,
        and build the foundation for humanoid robotics development.
      </>
    ),
  },
  {
    title: 'Simulation with Gazebo & Isaac',
    icon: '🎮',
    description: (
      <>
        Create digital twins, design URDF humanoid models, and leverage
        NVIDIA Isaac for high-fidelity physics simulation and training.
      </>
    ),
  },
  {
    title: 'Vision-Language-Action Systems',
    icon: '🧠',
    description: (
      <>
        Implement cutting-edge VLA architectures that combine vision, language
        understanding, and action generation for intelligent robots.
      </>
    ),
  },
];

function Feature({icon, title, description}) {
  return (
    <div className={clsx('col col--4')}>
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
    title: 'Module 1: Physical AI Foundations',
    link: '/docs/foundations/intro',
    description: 'Embodied intelligence, ROS 2 fundamentals, and core concepts.',
  },
  {
    title: 'Module 2: Digital Twins & Gazebo',
    link: '/docs/simulation/intro',
    description: 'Simulation environments, URDF modeling, and physics engines.',
  },
  {
    title: 'Module 3: Perception & NVIDIA Isaac',
    link: '/docs/perception/intro',
    description: 'Sensor fusion, sim-to-real transfer, and Isaac workflows.',
  },
  {
    title: 'Module 4: Vision-Language-Action',
    link: '/docs/vla-systems/intro',
    description: 'VLA architectures, LLM planning, and embodied reasoning.',
  },
  {
    title: 'Module 5: End-to-End Learning',
    link: '/docs/embodied-learning/intro',
    description: 'Imitation learning, RL, and diffusion models for robotics.',
  },
  {
    title: 'Module 6: Scaling & Production',
    link: '/docs/scaling-systems/intro',
    description: 'Distributed training, fleet management, and deployment.',
  },
  {
    title: 'Module 7: Capstone & Deployment',
    link: '/docs/capstone-deployment/intro',
    description: 'Production architecture, operations, and case studies.',
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
