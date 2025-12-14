"""
RL Policy for Robot Manipulation using PPO

Proximal Policy Optimization (PPO) for embodied learning.

Learning Goals:
- Understand policy gradient methods (REINFORCE → PPO)
- Learn actor-critic architecture (policy + value function)
- Implement PPO with clipping to prevent overshooting
- Train agents through trial-and-error in simulation

Key Concepts:
- Policy π(a|s): Maps observation → action distribution
- Value V(s): Predicts expected return from state
- Advantage: How much better than expected this trajectory is
- PPO clip: Prevent policy from changing too much per step

Real-World Application:
- Learn beyond demonstrations (superhuman performance)
- Exploration via trial-and-error
- Success: 90-98% after 2-4 weeks training
- Used by: OpenAI, DeepMind, Boston Dynamics

Training Process:
1. Collect trajectories from current policy
2. Compute advantages using value function
3. Update policy (with PPO clipping) to maximize advantage
4. Update value function to predict returns accurately
5. Repeat until convergence

Example:
    >>> policy = ActorCriticPolicy(obs_dim=10, action_dim=4)
    >>> env = RobotEnvironment()
    >>> for episode in range(10000):
    ...     trajectory = collect_episode(env, policy)
    ...     loss = ppo_loss(trajectory, policy, old_policy)
    ...     loss.backward()
    ...     optimizer.step()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import math


@dataclass
class Transition:
    """Single step in an episode."""
    state: torch.Tensor  # Observation
    action: torch.Tensor  # Action taken
    reward: float  # Immediate reward
    next_state: torch.Tensor  # Resulting observation
    done: bool  # Episode terminated?
    log_prob: float = 0.0  # Log probability of action under policy
    value: float = 0.0  # Value estimate for this state


@dataclass
class Trajectory:
    """Complete episode trajectory."""
    transitions: List[Transition]
    returns: torch.Tensor  # Cumulative discounted returns
    advantages: torch.Tensor  # Advantage estimates
    log_probs: torch.Tensor  # Log probabilities of actions


class ActorNetwork(nn.Module):
    """Policy network π(a|s) - predicts action distribution given observation."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 continuous: bool = True):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer size
            continuous: If True, output Gaussian; if False, output categorical
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous

        # Shared features
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if continuous:
            # Continuous action: output mean and log-std
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # Discrete action: output logits
            self.logits = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        """
        Args:
            obs: (batch_size, obs_dim)

        Returns:
            distribution: Gaussian (continuous) or Categorical (discrete)
        """
        features = self.features(obs)

        if self.continuous:
            mean = self.mean(features)
            std = torch.exp(self.log_std)
            distribution = Normal(mean, std)
        else:
            logits = self.logits(features)
            distribution = Categorical(logits=logits)

        return distribution

    def sample_action(self, obs: torch.Tensor,
                     deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: (batch_size, obs_dim) or (obs_dim,)
            deterministic: If True, return mean (no sampling)

        Returns:
            action: (batch_size, action_dim) or (action_dim,)
            log_prob: (batch_size,) or scalar - log probability
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        dist = self.forward(obs)

        if deterministic:
            if self.continuous:
                action = dist.mean
            else:
                action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        if squeeze:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)

        return action, log_prob


class CriticNetwork(nn.Module):
    """Value network V(s) - predicts expected return from observation."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        """
        Args:
            obs_dim: Observation dimension
            hidden_dim: Hidden layer size
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch_size, obs_dim)

        Returns:
            value: (batch_size, 1)
        """
        return self.net(obs)


class PPOBuffer:
    """Batch buffer for PPO algorithm."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, transition: Transition):
        """Add transition to buffer."""
        self.states.append(transition.state)
        self.actions.append(transition.action)
        self.rewards.append(transition.reward)
        self.next_states.append(transition.next_state)
        self.dones.append(transition.done)
        self.log_probs.append(transition.log_prob)
        self.values.append(transition.value)

    def compute_returns_advantages(self, gamma: float = 0.99,
                                   gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).

        Args:
            gamma: Discount factor
            gae_lambda: GAE parameter

        Returns:
            returns: (batch_size,)
            advantages: (batch_size,)
        """
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        # Compute TD residuals
        next_values = torch.cat([values[1:], torch.zeros(1)])
        deltas = rewards + gamma * next_values * (1 - dones) - values

        # Compute advantages via GAE
        advantages = torch.zeros_like(deltas)
        gae = 0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Compute returns
        returns = advantages + values

        return returns, advantages

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample mini-batch from buffer."""
        indices = np.random.choice(len(self.states), batch_size, replace=True)

        return {
            'states': torch.stack(self.states)[indices],
            'actions': torch.stack(self.actions)[indices],
            'log_probs_old': torch.tensor(self.log_probs)[indices],
        }


class PPOPolicy:
    """PPO agent for robot control."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 continuous: bool = True, learning_rate: float = 1e-4,
                 device: str = 'cuda'):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer size
            continuous: True for continuous actions, False for discrete
            learning_rate: Optimizer learning rate
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous

        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim,
                                 continuous).to(device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(device)

        # Target networks (for computing old log probs)
        self.actor_old = ActorNetwork(obs_dim, action_dim, hidden_dim,
                                      continuous).to(device)
        self.copy_weights()

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )

        # PPO hyperparameters
        self.clip_ratio = 0.2  # PPO clipping
        self.entropy_coef = 0.01  # Entropy bonus
        self.value_coef = 0.5  # Value loss weight

        self.buffer = PPOBuffer(capacity=10000)

    def copy_weights(self):
        """Copy actor weights to actor_old."""
        self.actor_old.load_state_dict(self.actor.state_dict())

    def select_action(self, obs: np.ndarray,
                     deterministic: bool = False) -> np.ndarray:
        """
        Select action from current policy.

        Args:
            obs: (obs_dim,) numpy array
            deterministic: If True, return mean (no sampling)

        Returns:
            action: (action_dim,) numpy array
        """
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample_action(obs_tensor, deterministic)
        return action.cpu().numpy()

    def compute_value(self, obs: np.ndarray) -> float:
        """
        Estimate value (expected return) from observation.

        Args:
            obs: (obs_dim,) numpy array

        Returns:
            value: scalar
        """
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            value = self.critic(obs_tensor)
        return value.item()

    def store_transition(self, transition: Transition):
        """Store transition in buffer."""
        self.buffer.add(transition)

    def train_step(self, batch_size: int = 64, epochs: int = 3) -> Dict[str, float]:
        """
        PPO training step.

        Args:
            batch_size: Mini-batch size
            epochs: Number of passes through data

        Returns:
            Dictionary with loss values
        """
        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_advantages()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Stack all data
        states = torch.stack(self.buffer.states).to(self.device)
        actions = torch.stack(self.buffer.actions).to(self.device)
        log_probs_old = torch.tensor(self.buffer.log_probs).to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # Training loop
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(epochs):
            # Sample mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for i in range(0, len(states), batch_size):
                batch_indices = indices[i:i + batch_size]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Actor update
                dist = self.actor(batch_states)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO ratio (probability ratio)
                ratio = torch.exp(log_probs - batch_log_probs_old)

                # PPO objective with clipping
                obj1 = ratio * batch_advantages
                obj2 = torch.clamp(ratio, 1 - self.clip_ratio,
                                  1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(obj1, obj2).mean()

                # Critic update
                values = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(values, batch_returns)

                # Total loss
                loss = (actor_loss + self.value_coef * critic_loss -
                       self.entropy_coef * entropy)

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) +
                    list(self.critic.parameters()),
                    max_norm=0.5
                )
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        # Update old policy
        self.copy_weights()

        # Clear buffer
        self.buffer.clear()

        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }


class SimpleEnvironment:
    """Minimal robot environment for testing."""

    def __init__(self, obs_dim: int = 10, action_dim: int = 4):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.state = np.random.randn(self.obs_dim).astype(np.float32) * 0.5
        self.steps = 0
        self.max_steps = 100
        return self.state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action in environment.

        Reward: How close to origin (simple reaching task)
        """
        self.state += action * 0.1  # Action affects state
        self.state += np.random.randn(self.obs_dim) * 0.01  # Environment noise
        self.state = np.clip(self.state, -1, 1)

        # Reward: negative distance from origin (reaching task)
        reward = -np.linalg.norm(self.state)

        self.steps += 1
        done = self.steps >= self.max_steps

        return self.state, reward, done


def collect_episode(env: SimpleEnvironment, policy: PPOPolicy) -> float:
    """Collect one episode using policy."""
    obs = env.reset()
    total_reward = 0

    while True:
        # Select action
        action = policy.select_action(obs)

        # Take step
        next_obs, reward, done = env.step(action)

        # Get value and log prob
        value = policy.compute_value(obs)
        action_tensor = torch.from_numpy(action).float()
        obs_tensor = torch.from_numpy(obs).float().to(policy.device)
        with torch.no_grad():
            _, log_prob = policy.actor.sample_action(obs_tensor, deterministic=False)

        # Store transition
        transition = Transition(
            state=torch.from_numpy(obs).float(),
            action=torch.from_numpy(action).float(),
            reward=reward,
            next_state=torch.from_numpy(next_obs).float(),
            done=done,
            log_prob=log_prob.item(),
            value=value,
        )
        policy.store_transition(transition)

        total_reward += reward
        obs = next_obs

        if done:
            break

    return total_reward


def main():
    """Example PPO training pipeline."""

    print("PPO Policy Training Example")
    print("=" * 50)

    # Hyperparameters
    obs_dim = 10
    action_dim = 4
    hidden_dim = 256
    num_episodes = 1000
    episodes_per_update = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

    # Environment and policy
    env = SimpleEnvironment(obs_dim, action_dim)
    policy = PPOPolicy(obs_dim, action_dim, hidden_dim, continuous=True,
                      device=device)

    print(f"Actor parameters: {sum(p.numel() for p in policy.actor.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in policy.critic.parameters()):,}")

    # Training loop
    print("\nTraining...")
    episode_rewards = []

    for episode in range(num_episodes):
        # Collect episodes
        for _ in range(episodes_per_update):
            reward = collect_episode(env, policy)
            episode_rewards.append(reward)

        # Train
        if len(policy.buffer.states) > 0:
            loss_dict = policy.train_step(batch_size=64, epochs=3)

        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-episodes_per_update*10:])
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.3f}, "
                  f"Actor Loss: {loss_dict['actor_loss']:.4f}, "
                  f"Critic Loss: {loss_dict['critic_loss']:.4f}")

    # Evaluation
    print("\nEvaluation...")
    test_rewards = []
    for _ in range(10):
        obs = env.reset()
        total_reward = 0
        for _ in range(100):
            action = policy.select_action(obs, deterministic=True)
            obs, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        test_rewards.append(total_reward)

    print(f"Test reward: {np.mean(test_rewards):.3f} ± {np.std(test_rewards):.3f}")

    print("\nPPO training complete!")


if __name__ == '__main__':
    main()
