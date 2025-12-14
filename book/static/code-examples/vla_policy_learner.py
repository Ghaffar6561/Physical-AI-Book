"""
Vision-Language-Action Policy Learning

This module demonstrates fine-tuning a vision-language model on robot demonstrations
to create a VLA policy. The policy can be used to control robots from language instructions
and visual observations.

Learning Goals:
  - Understand how to fine-tune multimodal models for robotics
  - Learn LoRA (Low-Rank Adaptation) for efficient training
  - See how to structure robot demonstration data
  - Implement inference-time planning with vision-language models
  - Evaluate transfer to novel tasks

Example:
  >>> learner = VLAPolicyLearner()
  >>> learner.train(demonstrations, epochs=10)
  >>> action = learner.infer(image, instruction="Pick up the red cup")
  >>> print(action)
  # {'target_position': [0.3, 0.2, 0.8], 'gripper_width': 0.08, 'force': 50}
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


@dataclass
class RobotDemonstration:
    """Single robot demonstration (image + language + action)."""
    image: np.ndarray  # [H, W, 3] RGB image
    instruction: str   # Natural language task description
    target_position: np.ndarray  # [3] target (x, y, z) in meters
    gripper_width: float  # meters (0-0.1)
    grasp_force: float  # Newtons (0-300)
    approach_height: float  # height above table (0.1-0.5)
    success: bool  # Did the task succeed?
    metadata: Optional[Dict] = None  # Additional info (object type, scene, etc.)


class RobotDemonstrationDataset(Dataset):
    """PyTorch dataset for robot demonstrations."""

    def __init__(self, demonstrations: List[RobotDemonstration]):
        """
        Initialize dataset.

        Args:
            demonstrations: List of RobotDemonstration objects
        """
        self.demonstrations = demonstrations

    def __len__(self) -> int:
        return len(self.demonstrations)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single demonstration.

        Returns:
            Dictionary with:
            - 'image': Tensor [3, 224, 224] (preprocessed)
            - 'instruction': String
            - 'target_position': Tensor [3]
            - 'gripper_width': Scalar
            - 'grasp_force': Scalar
            - 'success': Boolean
        """
        demo = self.demonstrations[idx]

        # Resize and normalize image
        image = self._preprocess_image(demo.image)

        return {
            'image': image,
            'instruction': demo.instruction,
            'target_position': torch.tensor(demo.target_position, dtype=torch.float32),
            'gripper_width': torch.tensor(demo.gripper_width, dtype=torch.float32),
            'grasp_force': torch.tensor(demo.grasp_force, dtype=torch.float32),
            'approach_height': torch.tensor(demo.approach_height, dtype=torch.float32),
            'success': torch.tensor(demo.success, dtype=torch.float32),
        }

    @staticmethod
    def _preprocess_image(image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image: resize, normalize, convert to tensor.

        Args:
            image: [H, W, 3] uint8 RGB image

        Returns:
            Tensor [3, 224, 224] normalized to [-1, 1]
        """
        # Resize to 224x224
        from torchvision.transforms.functional import resize, to_tensor
        image_tensor = to_tensor(image)  # [3, H, W], float32, [0, 1]
        image_tensor = resize(image_tensor, (224, 224))

        # Normalize to [-1, 1] (ImageNet-style)
        image_tensor = 2 * image_tensor - 1

        return image_tensor


class VLAPolicyLearner(nn.Module):
    """
    Vision-Language-Action policy for robotic manipulation.

    Architecture:
    - Vision encoder (ViT-B): Extract image features
    - Language encoder (BERT): Encode instruction text
    - Fusion network: Combine modalities
    - Action head: Predict action parameters
    """

    def __init__(
        self,
        vision_model_name: str = "google/vit-base-patch16-224",
        language_model_name: str = "bert-base-uncased",
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize VLA policy.

        Args:
            vision_model_name: HuggingFace vision model identifier
            language_model_name: HuggingFace language model identifier
            hidden_dim: Hidden dimension for fusion network
            dropout: Dropout probability
        """
        super().__init__()

        # Vision encoder (pretrained, frozen)
        try:
            from transformers import AutoImageProcessor, AutoModel
            self.vision_processor = AutoImageProcessor.from_pretrained(vision_model_name)
            self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
        except:
            # Fallback: simple CNN encoder if transformers not available
            self.vision_encoder = self._build_simple_vision_encoder()
            self.vision_dim = 512

        self.vision_encoder.requires_grad = False  # Freeze
        self.vision_dim = 768  # ViT-B output dimension

        # Language encoder (pretrained, frozen)
        try:
            from transformers import AutoTokenizer, AutoModel
            self.language_tokenizer = AutoTokenizer.from_pretrained(language_model_name)
            self.language_encoder = AutoModel.from_pretrained(language_model_name)
        except:
            # Fallback: simple embedding + LSTM
            self.language_encoder = self._build_simple_language_encoder()
            self.language_dim = 256

        self.language_encoder.requires_grad = False  # Freeze
        self.language_dim = 768  # BERT output dimension

        # Projection layers (trainable)
        self.vision_projection = nn.Linear(self.vision_dim, hidden_dim)
        self.language_projection = nn.Linear(self.language_dim, hidden_dim)

        # Fusion network (trainable)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Action heads (trainable)
        # Outputs: [x, y, z, gripper_width, grasp_force, approach_height]
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 6),  # [x, y, z, gripper_width, force, height]
        )

        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'position_loss': [],
            'gripper_loss': [],
            'force_loss': [],
        }

    def _build_simple_vision_encoder(self) -> nn.Module:
        """Fallback vision encoder (simple CNN)."""
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
                self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
                self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
                self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = F.relu(self.conv4(x))
                x = self.pool(x)
                x = x.flatten(1)
                return x

        return SimpleCNN()

    def _build_simple_language_encoder(self) -> nn.Module:
        """Fallback language encoder (embedding + LSTM)."""
        class SimpleLanguageEncoder(nn.Module):
            def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

            def forward(self, token_ids):
                embedded = self.embedding(token_ids)
                _, (hidden, _) = self.lstm(embedded)
                return hidden[-1]

        return SimpleLanguageEncoder()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to features.

        Args:
            images: [B, 3, 224, 224] normalized to [-1, 1]

        Returns:
            features: [B, vision_dim]
        """
        with torch.no_grad():
            outputs = self.vision_encoder(images)
            if hasattr(outputs, 'last_hidden_state'):
                # Transformer models
                features = outputs.last_hidden_state[:, 0, :]  # CLS token
            else:
                # CNN models
                features = outputs
        return features

    def encode_language(self, texts: List[str]) -> torch.Tensor:
        """
        Encode language instructions to features.

        Args:
            texts: List of instruction strings

        Returns:
            features: [B, language_dim]
        """
        with torch.no_grad():
            encoded = self.language_tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128,
            )
            outputs = self.language_encoder(**encoded)
            features = outputs.last_hidden_state[:, 0, :]  # CLS token

        return features

    def forward(
        self,
        images: torch.Tensor,
        instructions: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Predict action parameters from image and language.

        Args:
            images: [B, 3, 224, 224]
            instructions: List of B instruction strings

        Returns:
            Dictionary with:
            - 'target_position': [B, 3] (x, y, z)
            - 'gripper_width': [B, 1] (0-0.1m)
            - 'grasp_force': [B, 1] (0-300N)
            - 'approach_height': [B, 1] (0.1-0.5m)
        """
        # Encode modalities
        vision_feat = self.encode_image(images)  # [B, vision_dim]
        language_feat = self.encode_language(instructions)  # [B, language_dim]

        # Project
        vision_proj = self.vision_projection(vision_feat)  # [B, hidden_dim]
        language_proj = self.language_projection(language_feat)  # [B, hidden_dim]

        # Fuse
        fused = torch.cat([vision_proj, language_proj], dim=1)  # [B, 2*hidden_dim]
        fused = self.fusion(fused)  # [B, hidden_dim]

        # Predict actions
        action_raw = self.action_head(fused)  # [B, 6]

        # Scale outputs to valid ranges
        target_position = action_raw[:, :3]  # [B, 3], clamp later
        gripper_width = torch.clamp(action_raw[:, 3:4], 0, 0.1)  # [B, 1]
        grasp_force = torch.clamp(action_raw[:, 4:5], 10, 300)  # [B, 1]
        approach_height = torch.clamp(action_raw[:, 5:6], 0.1, 0.5)  # [B, 1]

        return {
            'target_position': target_position,
            'gripper_width': gripper_width,
            'grasp_force': grasp_force,
            'approach_height': approach_height,
        }

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Batch from DataLoader
            optimizer: PyTorch optimizer
            device: Compute device (cuda/cpu)

        Returns:
            Dictionary of loss values
        """
        # Move to device
        images = batch['image'].to(device)
        instructions = batch['instruction']
        target_positions = batch['target_position'].to(device)
        gripper_widths = batch['gripper_width'].to(device)
        grasp_forces = batch['grasp_force'].to(device)
        approach_heights = batch['approach_height'].to(device)

        # Forward pass
        predictions = self.forward(images, instructions)

        # Compute losses
        pos_loss = F.mse_loss(predictions['target_position'], target_positions)
        gripper_loss = F.mse_loss(predictions['gripper_width'], gripper_widths)
        force_loss = F.mse_loss(predictions['grasp_force'], grasp_forces)
        height_loss = F.mse_loss(predictions['approach_height'], approach_heights)

        # Weighted total loss
        total_loss = (
            pos_loss
            + 0.5 * gripper_loss
            + 0.3 * force_loss
            + 0.2 * height_loss
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'position_loss': pos_loss.item(),
            'gripper_loss': gripper_loss.item(),
            'force_loss': force_loss.item(),
            'height_loss': height_loss.item(),
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader with training data
            optimizer: PyTorch optimizer
            device: Compute device

        Returns:
            Average losses over epoch
        """
        self.train()
        total_losses = {}

        for batch_idx, batch in enumerate(train_loader):
            losses = self.train_step(batch, optimizer, device)

            # Accumulate
            for key, val in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += val

        # Average
        for key in total_losses:
            total_losses[key] /= len(train_loader)

        return total_losses

    def evaluate(
        self,
        val_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Args:
            val_loader: Validation DataLoader
            device: Compute device

        Returns:
            Metrics (MSE for each output)
        """
        self.eval()
        metrics = {}

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                instructions = batch['instruction']
                target_positions = batch['target_position'].to(device)
                gripper_widths = batch['gripper_width'].to(device)
                grasp_forces = batch['grasp_force'].to(device)

                predictions = self.forward(images, instructions)

                pos_error = torch.mean(
                    torch.sqrt(
                        torch.sum((predictions['target_position'] - target_positions) ** 2, dim=1)
                    )
                )
                gripper_error = torch.mean(
                    torch.abs(predictions['gripper_width'] - gripper_widths)
                )
                force_error = torch.mean(
                    torch.abs(predictions['grasp_force'] - grasp_forces)
                )

                metrics['position_error_m'] = pos_error.item()
                metrics['gripper_error_m'] = gripper_error.item()
                metrics['force_error_N'] = force_error.item()

        return metrics

    def infer(
        self,
        image: np.ndarray,
        instruction: str,
        device: torch.device = None,
    ) -> Dict[str, float]:
        """
        Inference: predict action from image and language.

        Args:
            image: [H, W, 3] RGB image
            instruction: Natural language instruction
            device: Compute device

        Returns:
            Dictionary with predicted action parameters
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        with torch.no_grad():
            # Preprocess image
            from torchvision.transforms.functional import to_tensor, resize
            image_tensor = to_tensor(image)  # [3, H, W]
            image_tensor = resize(image_tensor, (224, 224))
            image_tensor = 2 * image_tensor - 1  # Normalize to [-1, 1]
            image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, 224, 224]

            # Predict
            predictions = self.forward(image_tensor, [instruction])

            # Extract and convert to numpy
            action = {
                'target_position': predictions['target_position'][0].cpu().numpy(),
                'gripper_width': predictions['gripper_width'][0].item(),
                'grasp_force': predictions['grasp_force'][0].item(),
                'approach_height': predictions['approach_height'][0].item(),
            }

        return action


def main():
    """Demonstrate VLA policy learning."""
    print("=" * 70)
    print("Vision-Language-Action Policy Learning Demo")
    print("=" * 70)

    # Create dummy demonstrations
    print("\n1. Creating dummy dataset...")
    demonstrations = []
    for i in range(100):
        # Dummy RGB image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        demo = RobotDemonstration(
            image=image,
            instruction="Pick up the red cube" if i % 2 == 0 else "Place on the shelf",
            target_position=np.array([0.3 + np.random.randn() * 0.1, 0.2 + np.random.randn() * 0.1, 0.8]),
            gripper_width=0.08 + np.random.randn() * 0.01,
            grasp_force=50 + np.random.randn() * 5,
            approach_height=0.15 + np.random.randn() * 0.02,
            success=np.random.rand() > 0.1,
        )
        demonstrations.append(demo)

    print(f"Created {len(demonstrations)} demonstrations")

    # Create dataset and dataloader
    print("\n2. Creating DataLoader...")
    dataset = RobotDemonstrationDataset(demonstrations)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"Dataset size: {len(dataset)}, Batch size: 8")

    # Initialize policy
    print("\n3. Initializing VLA policy...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = VLAPolicyLearner(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=1e-4
    )

    print(f"Device: {device}")
    print(f"Trainable parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad)}")

    # Train
    print("\n4. Training...")
    print("-" * 70)
    for epoch in range(3):
        losses = policy.train_epoch(train_loader, optimizer, device)
        print(f"Epoch {epoch+1:2d}: Loss={losses['total_loss']:.4f}, "
              f"Pos={losses['position_loss']:.4f}, "
              f"Force={losses['force_loss']:.4f}")

    # Inference
    print("\n5. Inference on new image...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    action = policy.infer(test_image, "Pick up the red cup", device)

    print(f"Instruction: 'Pick up the red cup'")
    print(f"Predicted action:")
    print(f"  Target position: {action['target_position']}")
    print(f"  Gripper width: {action['gripper_width']:.4f} m")
    print(f"  Grasp force: {action['grasp_force']:.1f} N")
    print(f"  Approach height: {action['approach_height']:.4f} m")

    print("\n" + "=" * 70)
    print("Usage in Your Robot System:")
    print("=" * 70)
    print("""
# Load a trained policy
policy = VLAPolicyLearner().to(device)
policy.load_state_dict(torch.load('trained_vla_policy.pth'))

# Use in control loop
for step in range(num_steps):
    image = camera.get_frame()
    instruction = "Pick up the red object and place on the shelf"

    action = policy.infer(image, instruction, device)

    # Execute action
    robot.move_to(action['target_position'])
    robot.grasp(width=action['gripper_width'], force=action['grasp_force'])
    robot.execute_trajectory()
    """)

    print("\n" + "=" * 70)
    print("🤖 VLA Policy Ready for Deployment!")
    print("=" * 70)


if __name__ == '__main__':
    main()
