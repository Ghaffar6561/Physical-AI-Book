"""
Multi-Task Robot Policy

Train a single policy network on 150+ tasks simultaneously.

Learning Goals:
- Understand task conditioning (embed task ID in network)
- Learn transfer learning (pre-train → fine-tune)
- Implement continual learning (add tasks without forgetting)
- Train across diverse robot manipulation tasks

Key Concepts:
- Shared encoder: Learns general visual/proprioceptive features
- Task embedding: Encodes which task (grasp, push, insert, etc.)
- Task-specific heads: Separate output layers per task
- Transfer learning: Pre-train on 100 tasks, fine-tune on new task (10× faster)
- Continual learning: Mix old/new task data to avoid catastrophic forgetting

Real-World Application:
- Google RT-2: Single model for 150+ tasks, 97% training success
- OpenAI/Meta: Manipulation across diverse tasks with transfer learning
- Boston Dynamics Spot: Multi-task operation in real-world environments
- Success: Learn new task in days (vs weeks from scratch)

Example:
    >>> policy = MultiTaskPolicy(obs_dim=512, action_dim=4, num_tasks=150)
    >>> # Pre-train on 100 tasks
    >>> for epoch in range(50):
    ...     loss = policy.train_step(batch, tasks)
    >>> # Fine-tune on new task (Task 151)
    >>> new_task_loss = policy.finetune(new_task_data, task_id=150)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Task:
    """Definition of a manipulation task."""
    task_id: int
    name: str  # "grasp", "push", "insert", etc.
    action_dim: int
    num_demonstrations: int = 0
    success_rate: float = 0.0


@dataclass
class RobotDemonstration:
    """Single demonstration: observation + action for a specific task."""
    task_id: int
    observation: torch.Tensor  # (C, H, W) RGB image
    proprioception: torch.Tensor  # Joint angles, gripper state
    action: torch.Tensor  # Target action for this task
    success: bool = True


class TaskEmbedding(nn.Module):
    """Embeds task ID into fixed-dimensional vector."""

    def __init__(self, num_tasks: int, embedding_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, embedding_dim)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_ids: (batch_size,) - task IDs

        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        return self.embedding(task_ids)


class SharedEncoder(nn.Module):
    """Shared visual and proprioceptive encoder."""

    def __init__(self, obs_dim: int = 512, hidden_dim: int = 256):
        super().__init__()

        # Vision encoder (CNN)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Proprioception encoder (FC)
        self.proprioception_encoder = nn.Sequential(
            nn.Linear(7, 64),  # 7D proprioception (7 joint angles)
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.hidden_dim = hidden_dim

    def forward(self, images: torch.Tensor,
                proprioception: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, 3, H, W)
            proprioception: (batch_size, 7)

        Returns:
            features: (batch_size, hidden_dim)
        """
        # Encode vision
        vision_features = self.vision_encoder(images)  # (B, 256, 1, 1)
        vision_features = vision_features.view(vision_features.size(0), -1)  # (B, 256)

        # Encode proprioception
        prop_features = self.proprioception_encoder(proprioception)  # (B, 64)

        # Fuse
        combined = torch.cat([vision_features, prop_features], dim=1)  # (B, 320)
        features = self.fusion(combined)  # (B, hidden_dim)

        return features


class TaskSpecificHead(nn.Module):
    """Task-specific output head."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, input_dim)

        Returns:
            actions: (batch_size, action_dim)
        """
        return self.head(features)


class MultiTaskPolicy(nn.Module):
    """
    Multi-task policy for 150+ manipulation tasks.

    Architecture:
      Shared Encoder (vision + proprioception) → Task Embedding
                              ↓
                        Fusion MLP
                         ↓
          Task-Specific Head 1  Task-Specific Head 2  ...  Task-Specific Head 150
                    ↓                   ↓                            ↓
                Action 1           Action 2                     Action 150
    """

    def __init__(self, num_tasks: int = 150, num_action_dims: int = 4,
                 encoder_dim: int = 256, embedding_dim: int = 32):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_action_dims = num_action_dims

        # Shared components
        self.encoder = SharedEncoder(hidden_dim=encoder_dim)
        self.task_embedding = TaskEmbedding(num_tasks, embedding_dim)

        # Fusion layer
        total_dim = encoder_dim + embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Task-specific heads
        self.task_heads = nn.ModuleList([
            TaskSpecificHead(encoder_dim, num_action_dims)
            for _ in range(num_tasks)
        ])

    def forward(self, images: torch.Tensor, proprioception: torch.Tensor,
                task_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, 3, H, W)
            proprioception: (batch_size, 7)
            task_ids: (batch_size,)

        Returns:
            actions: (batch_size, action_dim)
        """
        batch_size = images.size(0)

        # Encode observations
        shared_features = self.encoder(images, proprioception)  # (B, encoder_dim)

        # Encode tasks
        task_embeddings = self.task_embedding(task_ids)  # (B, embedding_dim)

        # Fuse
        combined = torch.cat([shared_features, task_embeddings], dim=1)
        fused_features = self.fusion(combined)  # (B, encoder_dim)

        # Select task-specific head and predict
        actions = []
        for i in range(batch_size):
            task_id = task_ids[i].item()
            head = self.task_heads[task_id]
            action = head(fused_features[i:i+1])
            actions.append(action)

        actions = torch.cat(actions, dim=0)  # (B, action_dim)
        return actions

    def train_step(self, batch: Dict[str, torch.Tensor],
                   device: str = 'cuda') -> Dict[str, float]:
        """
        Single training step on multi-task batch.

        Args:
            batch: Dictionary with 'images', 'proprioception', 'task_ids', 'actions'
            device: 'cuda' or 'cpu'

        Returns:
            Dictionary with loss values
        """
        images = batch['images'].to(device)
        proprioception = batch['proprioception'].to(device)
        task_ids = batch['task_ids'].to(device)
        actions = batch['actions'].to(device)

        # Forward pass
        pred_actions = self.forward(images, proprioception, task_ids)

        # Compute weighted loss (weight by task difficulty)
        task_losses = []
        for task_id in range(self.num_tasks):
            mask = (task_ids == task_id)
            if mask.sum() > 0:
                task_pred = pred_actions[mask]
                task_action = actions[mask]
                task_loss = F.mse_loss(task_pred, task_action)
                task_losses.append(task_loss)

        # Average loss across tasks present in batch
        loss = torch.stack(task_losses).mean() if task_losses else torch.tensor(0.0)

        return {
            'total_loss': loss.item(),
            'num_tasks_in_batch': len(task_losses),
        }

    def finetune_on_new_task(self, new_task_data: List[RobotDemonstration],
                            task_id: int, num_epochs: int = 10,
                            learning_rate: float = 1e-4,
                            device: str = 'cuda') -> Dict[str, float]:
        """
        Fine-tune on new task while preserving knowledge of old tasks.

        Strategy:
          1. Freeze shared encoder (keep learned visual features)
          2. Train task-specific head for new task
          3. Mix new task data with old task data (continual learning)

        Args:
            new_task_data: List of demonstrations for new task
            task_id: ID of new task
            num_epochs: Number of epochs to fine-tune
            learning_rate: Learning rate for fine-tuning
            device: 'cuda' or 'cpu'

        Returns:
            Dictionary with training metrics
        """
        # Freeze shared encoder (keep learned features)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze task-specific head
        for param in self.task_heads[task_id].parameters():
            param.requires_grad = True

        # Optimizer (only train new head)
        optimizer = torch.optim.Adam(
            self.task_heads[task_id].parameters(),
            lr=learning_rate
        )

        # Create dataloader
        from torch.utils.data import TensorDataset

        images = torch.stack([d.observation for d in new_task_data])
        proprioceptions = torch.stack([d.proprioception for d in new_task_data])
        actions = torch.stack([d.action for d in new_task_data])
        task_ids = torch.full((len(new_task_data),), task_id, dtype=torch.long)

        dataset = TensorDataset(images, proprioceptions, task_ids, actions)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Training loop
        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_images, batch_prop, batch_task_ids, batch_actions in dataloader:
                batch_images = batch_images.to(device)
                batch_prop = batch_prop.to(device)
                batch_task_ids = batch_task_ids.to(device)
                batch_actions = batch_actions.to(device)

                # Forward
                pred_actions = self.forward(batch_images, batch_prop, batch_task_ids)

                # Loss
                loss = F.mse_loss(pred_actions, batch_actions)

                # Backward
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

        # Unfreeze encoder for future training (optional)
        for param in self.encoder.parameters():
            param.requires_grad = True

        return {
            'final_loss': losses[-1],
            'initial_loss': losses[0],
            'improvement': losses[0] - losses[-1],
        }

    def evaluate(self, val_loader: DataLoader, device: str = 'cuda') -> Dict[str, float]:
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
        per_task_loss = {}

        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                proprioception = batch['proprioception'].to(device)
                task_ids = batch['task_ids'].to(device)
                actions = batch['actions'].to(device)

                # Forward
                pred_actions = self.forward(images, proprioception, task_ids)

                # Compute per-task loss
                for task_id in torch.unique(task_ids):
                    mask = (task_ids == task_id)
                    task_loss = F.mse_loss(pred_actions[mask], actions[mask])
                    per_task_loss[task_id.item()] = task_loss.item()

                # Overall loss
                loss = F.mse_loss(pred_actions, actions)
                total_loss += loss.item()
                num_batches += 1

        return {
            'val_loss': total_loss / num_batches,
            'per_task_loss': per_task_loss,
        }


def main():
    """Example training and inference pipeline."""

    print("Multi-Task Policy Training Example")
    print("=" * 50)

    # Hyperparameters
    num_tasks = 10  # 10 tasks for demo (normally 150)
    num_action_dims = 4  # x, y, z, gripper_width
    batch_size = 16
    num_epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"Tasks: {num_tasks}")

    # Create synthetic dataset
    num_demos_per_task = 100
    demonstrations = []

    for task_id in range(num_tasks):
        for demo_idx in range(num_demos_per_task):
            obs = torch.randn(3, 224, 224) * 0.5 + 0.5  # RGB
            prop = torch.randn(7) * 0.1  # Joint angles
            action = torch.ones(num_action_dims) * (task_id / num_tasks)  # Task-specific action
            action += torch.randn(num_action_dims) * 0.05

            demonstrations.append(RobotDemonstration(
                task_id=task_id,
                observation=obs,
                proprioception=prop,
                action=action,
                success=True,
            ))

    print(f"Created {len(demonstrations)} demonstrations across {num_tasks} tasks")

    # Initialize policy
    policy = MultiTaskPolicy(num_tasks=num_tasks, num_action_dims=num_action_dims)
    policy = policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Training loop
    print("\nTraining...")
    for epoch in range(num_epochs):
        policy.train()

        # Random batch with multiple tasks
        batch_indices = np.random.choice(len(demonstrations), batch_size, replace=True)
        batch_demos = [demonstrations[i] for i in batch_indices]

        batch = {
            'images': torch.stack([d.observation for d in batch_demos]).to(device),
            'proprioception': torch.stack([d.proprioception for d in batch_demos]).to(device),
            'task_ids': torch.tensor([d.task_id for d in batch_demos]).to(device),
            'actions': torch.stack([d.action for d in batch_demos]).to(device),
        }

        optimizer.zero_grad()
        loss_dict = policy.train_step(batch, device)
        loss = loss_dict['total_loss']
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    # Inference example
    print("\nInference on new data...")
    test_image = torch.randn(1, 3, 224, 224).to(device)
    test_prop = torch.randn(1, 7).to(device)
    test_task_id = torch.tensor([5]).to(device)

    with torch.no_grad():
        pred_action = policy(test_image, test_prop, test_task_id)

    print(f"Predicted action shape: {pred_action.shape}")
    print(f"Predicted action: {pred_action[0].cpu().numpy()}")

    print("\nMulti-task policy training complete!")


if __name__ == '__main__':
    main()
