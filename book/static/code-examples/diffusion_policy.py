"""
Diffusion Policy for Robot Manipulation

A complete implementation of trajectory generation via diffusion models.

Learning Goals:
- Understand how diffusion models generate robot trajectories
- Learn reverse process (denoising) for action generation
- Implement conditioning on observations
- Compare diffusion vs behavioral cloning

Key Concepts:
- Forward process: Add noise to trajectory → pure Gaussian
- Reverse process: Remove noise iteratively → clean trajectory
- Conditioning: Use observation to guide denoising
- Multimodal actions: One observation → multiple valid trajectories

Real-World Application:
- Handles multiple valid grasping angles
- Generates smooth, natural motion
- Better generalization than behavioral cloning
- Success: 80-90% on manipulation tasks (vs BC: 50-70%)

Example:
    >>> policy = DiffusionPolicy()
    >>> observation = torch.randn(1, 3, 224, 224)  # RGB image
    >>> trajectory = policy.infer(observation, num_steps=50)
    >>> robot.execute_trajectory(trajectory)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass
import math


@dataclass
class RobotDemonstration:
    """Single robot trajectory demonstration."""
    observation: torch.Tensor  # (T, C, H, W): T frames of RGB images
    action_trajectory: torch.Tensor  # (T, action_dim): Actions over time
    task_id: int = 0  # Which task (grasp, push, insert, etc.)
    success: bool = True
    object_class: str = "unknown"


class DemonstrationDataset(Dataset):
    """Dataset for diffusion policy training."""

    def __init__(self, demonstrations: List[RobotDemonstration],
                 trajectory_length: int = 10,
                 action_dim: int = 4):
        """
        Args:
            demonstrations: List of RobotDemonstration objects
            trajectory_length: How many steps to predict ahead
            action_dim: Dimension of action space (e.g., 4 for x,y,z,gripper)
        """
        self.demonstrations = demonstrations
        self.trajectory_length = trajectory_length
        self.action_dim = action_dim

    def __len__(self):
        return len(self.demonstrations)

    def __getitem__(self, idx):
        demo = self.demonstrations[idx]

        # Get observation (take first frame)
        observation = demo.observation[0]  # (C, H, W)

        # Normalize to [-1, 1]
        if observation.max() > 1:
            observation = observation / 127.5 - 1.0

        # Get trajectory (pad or truncate to trajectory_length)
        trajectory = demo.action_trajectory
        if trajectory.shape[0] < self.trajectory_length:
            # Pad with last action
            padding = self.trajectory_length - trajectory.shape[0]
            trajectory = torch.cat([
                trajectory,
                trajectory[-1:].repeat(padding, 1)
            ], dim=0)
        else:
            # Take first trajectory_length steps
            trajectory = trajectory[:self.trajectory_length]

        return {
            'observation': observation,
            'trajectory': trajectory,
        }


class NoiseSchedule:
    """Variance schedule for diffusion process."""

    def __init__(self, num_steps: int = 1000, schedule_type: str = 'linear'):
        """
        Args:
            num_steps: Total number of diffusion steps
            schedule_type: 'linear' or 'cosine'
        """
        self.num_steps = num_steps

        if schedule_type == 'linear':
            # Linear schedule
            betas = torch.linspace(0.0001, 0.02, num_steps)
        elif schedule_type == 'cosine':
            # Cosine schedule (more sophisticated)
            s = 0.008
            steps = torch.arange(num_steps + 1)
            alphas_cumprod = torch.cos(
                ((steps / num_steps) + s) / (1 + s) * math.pi * 0.5
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Pre-compute useful terms
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Variance terms
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                           torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                           torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                           torch.sqrt(1.0 / alphas_cumprod - 1))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Store tensor as buffer (not trainable)."""
        setattr(self, name, tensor)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                noise: torch.Tensor) -> torch.Tensor:
        """
        Forward process: Add noise to x_0 at timestep t.

        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

        Args:
            x_0: Clean trajectory (batch_size, trajectory_length, action_dim)
            t: Timestep (batch_size,)
            noise: Gaussian noise (same shape as x_0)

        Returns:
            x_t: Noisy trajectory at step t
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]  # (batch_size,)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return (sqrt_alphas_cumprod_t * x_0 +
                sqrt_one_minus_alphas_cumprod_t * noise)


class DiffusionUNet(nn.Module):
    """
    Simplified U-Net for diffusion model.

    Predicts noise in trajectory given:
    - Noisy trajectory
    - Timestep embedding
    - Observation conditioning
    """

    def __init__(self, action_dim: int = 4, trajectory_length: int = 10,
                 hidden_dim: int = 256):
        super().__init__()

        self.action_dim = action_dim
        self.trajectory_length = trajectory_length
        self.hidden_dim = hidden_dim

        # Input: flattened trajectory (trajectory_length * action_dim,)
        trajectory_input_dim = trajectory_length * action_dim

        # Observation encoder (image → features)
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        obs_feature_dim = 128

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main denoising network
        total_input_dim = trajectory_input_dim + obs_feature_dim + hidden_dim

        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, trajectory_input_dim),  # Output: noise
        )

    def forward(self, trajectory: torch.Tensor, timestep: torch.Tensor,
                observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: (batch_size, trajectory_length, action_dim)
            timestep: (batch_size,) or (batch_size, 1) - timestep indices
            observation: (batch_size, 3, H, W) - RGB image

        Returns:
            noise_pred: (batch_size, trajectory_length, action_dim)
        """
        batch_size = trajectory.shape[0]

        # Flatten trajectory
        traj_flat = trajectory.view(batch_size, -1)  # (batch_size, T*action_dim)

        # Encode observation
        obs_features = self.obs_encoder(observation)  # (batch_size, 128, 1, 1)
        obs_features = obs_features.view(batch_size, -1)  # (batch_size, 128)

        # Embed timestep (normalize to 0-1)
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(-1).float()
        timestep_norm = timestep / 1000.0  # Normalize to ~[0, 1]
        time_features = self.time_embed(timestep_norm)  # (batch_size, hidden_dim)

        # Concatenate all inputs
        combined = torch.cat([traj_flat, obs_features, time_features], dim=1)

        # Predict noise
        noise_pred_flat = self.net(combined)  # (batch_size, T*action_dim)

        # Reshape back to trajectory shape
        noise_pred = noise_pred_flat.view(batch_size, self.trajectory_length,
                                          self.action_dim)

        return noise_pred


class DiffusionPolicy(nn.Module):
    """Complete diffusion policy for trajectory generation."""

    def __init__(self, action_dim: int = 4, trajectory_length: int = 10,
                 num_diffusion_steps: int = 1000, hidden_dim: int = 256):
        """
        Args:
            action_dim: Dimension of action space
            trajectory_length: Number of steps to predict
            num_diffusion_steps: Steps in diffusion process (training)
            hidden_dim: Hidden dimension of U-Net
        """
        super().__init__()

        self.action_dim = action_dim
        self.trajectory_length = trajectory_length
        self.num_diffusion_steps = num_diffusion_steps

        # Noise schedule
        self.noise_schedule = NoiseSchedule(num_diffusion_steps, 'cosine')

        # Denoising network
        self.denoise_net = DiffusionUNet(action_dim, trajectory_length, hidden_dim)

    def forward(self, observation: torch.Tensor, trajectory: torch.Tensor,
                timestep: torch.Tensor) -> torch.Tensor:
        """
        Predict noise in trajectory at given timestep.

        Args:
            observation: (batch_size, 3, H, W) - RGB image
            trajectory: (batch_size, trajectory_length, action_dim) - noisy trajectory
            timestep: (batch_size,) - timestep indices

        Returns:
            noise_pred: (batch_size, trajectory_length, action_dim)
        """
        return self.denoise_net(trajectory, timestep, observation)

    def train_step(self, batch: Dict[str, torch.Tensor],
                  device: str = 'cuda') -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Dictionary with 'observation' and 'trajectory'
            device: 'cuda' or 'cpu'

        Returns:
            Dictionary with loss values
        """
        observation = batch['observation'].to(device)  # (B, 3, H, W)
        trajectory = batch['trajectory'].to(device)  # (B, T, action_dim)

        batch_size = trajectory.shape[0]

        # Sample random timestep for each sample
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,),
                         device=device)

        # Sample random noise
        noise = torch.randn_like(trajectory)

        # Forward process: add noise to trajectory
        x_t = self.noise_schedule.q_sample(trajectory, t, noise)

        # Predict noise using network
        noise_pred = self(observation, x_t, t)

        # MSE loss on noise prediction
        loss = F.mse_loss(noise_pred, noise)

        return {
            'total_loss': loss.item(),
            'noise_mse': loss.item(),
        }

    def infer(self, observation: torch.Tensor, num_steps: int = 50,
             device: str = 'cuda') -> torch.Tensor:
        """
        Generate trajectory via iterative denoising.

        Args:
            observation: (1, 3, H, W) - Single RGB image
            num_steps: Number of denoising steps (50-100 typical)
            device: 'cuda' or 'cpu'

        Returns:
            trajectory: (1, trajectory_length, action_dim)
        """
        observation = observation.to(device)

        # Start with random noise
        x = torch.randn(1, self.trajectory_length, self.action_dim,
                       device=device)

        # Denoising loop
        for step in reversed(range(0, self.num_diffusion_steps,
                                   self.num_diffusion_steps // num_steps)):
            # Get current timestep
            t = torch.tensor([step], device=device)

            # Predict noise
            with torch.no_grad():
                noise_pred = self(observation, x, t)

            # Update x using DDPM equation
            alpha_t = self.noise_schedule.alphas_cumprod[step].item()
            alpha_prev = self.noise_schedule.alphas_cumprod_prev[step].item()

            # Reverse step
            if step > 0:
                sigma = math.sqrt((1 - alpha_prev) / (1 - alpha_t) *
                                 (1 - alpha_t / alpha_prev))
                z = torch.randn_like(x)
            else:
                sigma = 0
                z = 0

            # DDPM reverse formula
            c1 = 1 / math.sqrt(self.noise_schedule.alphas[step].item())
            c2 = (1 - self.noise_schedule.alphas[step].item()) / math.sqrt(
                1 - alpha_t)

            x = c1 * (x - c2 * noise_pred) + sigma * z

        # Clamp to [-1, 1]
        x = torch.clamp(x, -1, 1)

        return x

    def evaluate(self, val_loader: DataLoader,
                device: str = 'cuda') -> Dict[str, float]:
        """
        Evaluate policy on validation set.

        Args:
            val_loader: Validation DataLoader
            device: 'cuda' or 'cpu'

        Returns:
            Dictionary with evaluation metrics
        """
        self.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                loss_dict = self.train_step(batch, device)
                total_loss += loss_dict['total_loss']
                num_batches += 1

        return {
            'val_loss': total_loss / num_batches,
        }


def main():
    """Example training and inference pipeline."""

    print("Diffusion Policy Training Example")
    print("=" * 50)

    # Hyperparameters
    action_dim = 4  # x, y, z, gripper_width
    trajectory_length = 10  # Predict 10 steps ahead
    batch_size = 16
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")

    # Create synthetic dataset
    num_demos = 100
    demonstrations = []
    for i in range(num_demos):
        # Synthetic trajectory
        obs = torch.randn(1, 3, 224, 224) * 0.5 + 0.5  # RGB [0,1]

        # Generate smooth trajectory
        trajectory = torch.linspace(0, 1, trajectory_length).unsqueeze(-1)
        trajectory = trajectory.repeat(1, action_dim) * 0.5 + 0.25
        # Add small noise for realism
        trajectory += torch.randn_like(trajectory) * 0.05
        trajectory = torch.clamp(trajectory, -1, 1)

        demonstrations.append(
            RobotDemonstration(
                observation=obs,
                action_trajectory=trajectory,
                success=True,
            )
        )

    print(f"Created {len(demonstrations)} synthetic demonstrations")

    # Create dataset and loader
    dataset = DemonstrationDataset(demonstrations, trajectory_length, action_dim)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize policy
    policy = DiffusionPolicy(action_dim, trajectory_length,
                           num_diffusion_steps=1000).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Training loop
    print("\nTraining...")
    for epoch in range(num_epochs):
        policy.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss_dict = policy.train_step(batch, device)
            loss = loss_dict['total_loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss

        avg_loss = epoch_loss / len(train_loader)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Inference example
    print("\nInference...")
    policy.eval()

    test_obs = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        trajectory = policy.infer(test_obs, num_steps=50, device=device)

    print(f"Generated trajectory shape: {trajectory.shape}")
    print(f"Trajectory sample:\n{trajectory[0, :3, :].cpu().numpy()}")

    print("\nDiffusion Policy training complete!")


if __name__ == '__main__':
    main()
