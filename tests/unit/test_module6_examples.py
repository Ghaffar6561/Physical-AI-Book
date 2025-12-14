"""
Comprehensive Test Suite for Module 6 - Scaling Systems Code Examples

Tests for:
- multi_task_policy.py: Multi-task learning architecture
- fleet_manager.py: Distributed fleet orchestration

Test Coverage:
- Syntax validation
- Architecture structure
- Learning concepts validation
- Integration testing
- Documentation completeness
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from datetime import datetime
import inspect

# Add code-examples to path
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__),
    '../../book/static/code-examples'
))

# Import modules under test
try:
    from multi_task_policy import (
        MultiTaskPolicy, TaskEmbedding, SharedEncoder,
        TaskSpecificHead, Task, RobotDemonstration
    )
    MULTI_TASK_IMPORT_OK = True
except ImportError as e:
    MULTI_TASK_IMPORT_OK = False
    IMPORT_ERROR = str(e)

try:
    from fleet_manager import (
        FleetManager, RobotNode, CentralServer,
        FederatedLearningAggregator, RobotStats, ModelCheckpoint
    )
    FLEET_IMPORT_OK = True
except ImportError as e:
    FLEET_IMPORT_OK = False
    FLEET_IMPORT_ERROR = str(e)


class TestModule6ExampleSyntax:
    """Validate Python syntax and imports."""

    def test_multi_task_policy_imports(self):
        """Multi-task policy module imports successfully."""
        assert MULTI_TASK_IMPORT_OK, f"Import failed: {IMPORT_ERROR}"

    def test_fleet_manager_imports(self):
        """Fleet manager module imports successfully."""
        assert FLEET_IMPORT_OK, f"Import failed: {FLEET_IMPORT_ERROR}"

    def test_multi_task_policy_has_main(self):
        """Multi-task policy has main() function."""
        import multi_task_policy
        assert hasattr(multi_task_policy, 'main')
        assert callable(multi_task_policy.main)

    def test_fleet_manager_has_main(self):
        """Fleet manager has main() function."""
        import fleet_manager
        assert hasattr(fleet_manager, 'main')
        assert callable(fleet_manager.main)


class TestMultiTaskPolicyStructure:
    """Validate multi-task policy architecture."""

    def test_task_embedding_class_exists(self):
        """TaskEmbedding class is defined."""
        assert hasattr(torch.nn, 'Module')
        assert issubclass(TaskEmbedding, nn.Module)

    def test_task_embedding_initialization(self):
        """TaskEmbedding initializes with correct parameters."""
        embedding = TaskEmbedding(num_tasks=150, embedding_dim=32)
        assert embedding is not None
        assert hasattr(embedding, 'embedding')

    def test_task_embedding_forward(self):
        """TaskEmbedding forward pass works correctly."""
        embedding = TaskEmbedding(num_tasks=10, embedding_dim=32)
        task_ids = torch.tensor([0, 1, 2, 3])
        output = embedding(task_ids)
        assert output.shape == (4, 32)

    def test_shared_encoder_class_exists(self):
        """SharedEncoder class is defined."""
        assert issubclass(SharedEncoder, nn.Module)

    def test_shared_encoder_initialization(self):
        """SharedEncoder initializes correctly."""
        encoder = SharedEncoder(obs_dim=512, hidden_dim=256)
        assert hasattr(encoder, 'vision_encoder')
        assert hasattr(encoder, 'proprioception_encoder')
        assert hasattr(encoder, 'fusion')

    def test_shared_encoder_forward(self):
        """SharedEncoder forward pass produces correct output shape."""
        encoder = SharedEncoder(hidden_dim=256)
        images = torch.randn(4, 3, 224, 224)
        proprioception = torch.randn(4, 7)
        output = encoder(images, proprioception)
        assert output.shape == (4, 256)

    def test_task_specific_head_forward(self):
        """TaskSpecificHead forward pass works."""
        head = TaskSpecificHead(input_dim=256, action_dim=4)
        features = torch.randn(4, 256)
        actions = head(features)
        assert actions.shape == (4, 4)

    def test_multi_task_policy_initialization(self):
        """MultiTaskPolicy initializes with correct components."""
        policy = MultiTaskPolicy(num_tasks=150, num_action_dims=4)
        assert hasattr(policy, 'encoder')
        assert hasattr(policy, 'task_embedding')
        assert hasattr(policy, 'fusion')
        assert hasattr(policy, 'task_heads')
        assert len(policy.task_heads) == 150

    def test_multi_task_policy_forward(self):
        """MultiTaskPolicy forward pass produces actions."""
        policy = MultiTaskPolicy(num_tasks=10, num_action_dims=4)
        images = torch.randn(4, 3, 224, 224)
        proprioception = torch.randn(4, 7)
        task_ids = torch.tensor([0, 1, 2, 3])
        actions = policy(images, proprioception, task_ids)
        assert actions.shape == (4, 4)

    def test_multi_task_policy_train_step(self):
        """Train step returns loss dictionary."""
        policy = MultiTaskPolicy(num_tasks=5, num_action_dims=4)
        batch = {
            'images': torch.randn(4, 3, 224, 224),
            'proprioception': torch.randn(4, 7),
            'task_ids': torch.tensor([0, 1, 2, 3]),
            'actions': torch.randn(4, 4),
        }
        loss_dict = policy.train_step(batch, device='cpu')
        assert isinstance(loss_dict, dict)
        assert 'total_loss' in loss_dict
        assert 'num_tasks_in_batch' in loss_dict

    def test_multi_task_policy_finetune(self):
        """Finetune method works on new task."""
        policy = MultiTaskPolicy(num_tasks=10, num_action_dims=4)
        new_task_data = [
            RobotDemonstration(
                task_id=9,
                observation=torch.randn(3, 224, 224),
                proprioception=torch.randn(7),
                action=torch.randn(4),
                success=True,
            )
            for _ in range(10)
        ]
        result = policy.finetune_on_new_task(
            new_task_data, task_id=9,
            num_epochs=2, device='cpu'
        )
        assert isinstance(result, dict)
        assert 'final_loss' in result
        assert 'initial_loss' in result

    def test_multi_task_policy_evaluate(self):
        """Evaluate method computes validation metrics."""
        from torch.utils.data import DataLoader, TensorDataset

        policy = MultiTaskPolicy(num_tasks=5, num_action_dims=4)
        dataset = TensorDataset(
            torch.randn(20, 3, 224, 224),
            torch.randn(20, 7),
            torch.tensor([i % 5 for i in range(20)]),
            torch.randn(20, 4),
        )
        val_loader = DataLoader(dataset, batch_size=4)
        result = policy.evaluate(val_loader, device='cpu')
        assert isinstance(result, dict)
        assert 'val_loss' in result


class TestFleetManagerStructure:
    """Validate fleet manager architecture."""

    def test_robot_stats_dataclass(self):
        """RobotStats dataclass is properly defined."""
        stats = RobotStats(robot_id=0)
        assert stats.robot_id == 0
        assert stats.success_rate == 0.0
        assert isinstance(stats.failures, list)

    def test_model_checkpoint_dataclass(self):
        """ModelCheckpoint dataclass is properly defined."""
        weights = {'param1': torch.randn(10, 10)}
        checkpoint = ModelCheckpoint(
            version=1,
            timestamp=datetime.now(),
            weights=weights,
            training_loss=0.5,
            validation_accuracy=0.85,
            tasks_included=150,
        )
        assert checkpoint.version == 1
        assert checkpoint.validation_accuracy == 0.85

    def test_robot_node_initialization(self):
        """RobotNode initializes with correct attributes."""
        policy = nn.Linear(10, 4)
        robot = RobotNode(robot_id=0, policy=policy)
        assert robot.robot_id == 0
        assert robot.is_healthy is True
        assert robot.consecutive_failures == 0

    def test_robot_node_run_trial(self):
        """RobotNode can run a trial."""
        policy = nn.Sequential(nn.Linear(7 + 256, 4))
        robot = RobotNode(robot_id=0, policy=policy)
        obs = torch.randn(3, 224, 224)
        prop = torch.randn(7)
        result = robot.run_trial(task_id=0, observation=obs,
                                 proprioception=prop)
        assert isinstance(result, dict)
        assert 'success' in result

    def test_robot_node_collect_daily_data(self):
        """RobotNode collects daily data."""
        policy = nn.Sequential(nn.Linear(7 + 256, 4))
        robot = RobotNode(robot_id=0, policy=policy)
        robot.collect_daily_data(num_trials=100)
        assert len(robot.local_data_buffer) == 100

    def test_robot_node_upload_data(self):
        """RobotNode can upload data to server."""
        policy = nn.Sequential(nn.Linear(7 + 256, 4))
        robot = RobotNode(robot_id=0, policy=policy)
        robot.collect_daily_data(num_trials=50)

        server = CentralServer()
        success = robot.upload_data(server)
        assert success is True
        assert len(robot.local_data_buffer) == 0

    def test_robot_node_download_model(self):
        """RobotNode can download model checkpoint."""
        policy = nn.Sequential(nn.Linear(10, 4))
        robot = RobotNode(robot_id=0, policy=policy)

        checkpoint = ModelCheckpoint(
            version=1,
            timestamp=datetime.now(),
            weights=policy.state_dict(),
            training_loss=0.5,
            validation_accuracy=0.85,
            tasks_included=10,
        )
        success = robot.download_model(checkpoint)
        assert success is True
        assert robot.stats.model_version == 1

    def test_robot_node_health_status(self):
        """RobotNode reports health status."""
        policy = nn.Sequential(nn.Linear(10, 4))
        robot = RobotNode(robot_id=5, policy=policy)
        health = robot.get_health_status()
        assert health['robot_id'] == 5
        assert 'is_healthy' in health
        assert 'success_rate' in health

    def test_federated_learning_aggregator_init(self):
        """FederatedLearningAggregator initializes."""
        aggregator = FederatedLearningAggregator(num_robots=100)
        assert aggregator.num_robots == 100

    def test_federated_learning_aggregation(self):
        """FederatedLearningAggregator aggregates weights correctly."""
        aggregator = FederatedLearningAggregator(num_robots=3)

        global_weights = {'w': torch.ones(2, 2)}

        # Simulate 3 robots submitting updates
        for i in range(3):
            weights = {'w': torch.ones(2, 2) * (i + 1)}
            aggregator.collect_local_updates(robot_id=i, weights=weights)

        # Aggregate: should be (1 + 2 + 3) / 3 = 2.0
        result = aggregator.aggregate(global_weights)
        expected = torch.ones(2, 2) * 2.0
        assert torch.allclose(result['w'], expected)

    def test_central_server_initialization(self):
        """CentralServer initializes."""
        server = CentralServer(num_gpus=4)
        assert server.num_gpus == 4
        assert server.total_data_received == 0

    def test_central_server_receive_data(self):
        """CentralServer receives robot data."""
        server = CentralServer()
        data = [
            {'task_id': 0, 'success': True},
            {'task_id': 1, 'success': False},
        ]
        server.receive_robot_data(robot_id=0, data=data, size_mb=10.0)
        assert server.total_data_received == 10.0
        assert len(server.data_lake[0]) == 1

    def test_central_server_training(self):
        """CentralServer can train on data."""
        server = CentralServer()
        data = [{'task_id': 0, 'success': True}] * 100
        server.receive_robot_data(robot_id=0, data=data, size_mb=10.0)

        model = nn.Sequential(nn.Linear(10, 4))
        checkpoint = server.start_training(model, num_epochs=1)

        assert checkpoint.version == 1
        assert checkpoint.validation_accuracy >= 0.75

    def test_central_server_stats(self):
        """CentralServer reports statistics."""
        server = CentralServer()
        stats = server.get_server_stats()
        assert isinstance(stats, dict)
        assert 'total_data_received_mb' in stats
        assert 'current_model_version' in stats

    def test_fleet_manager_initialization(self):
        """FleetManager initializes with robots."""
        fleet = FleetManager(num_robots=10, num_tasks=20)
        assert len(fleet.robots) == 10
        assert fleet.central_server is not None
        assert fleet.aggregator is not None

    def test_fleet_manager_data_collection(self):
        """FleetManager coordinates data collection."""
        fleet = FleetManager(num_robots=5, num_tasks=10)
        fleet.start_data_collection(duration_days=1)

        # Check that robots collected data
        total_trials = sum(r.stats.trials_completed for r in fleet.robots)
        assert total_trials > 0

    def test_fleet_manager_sync_and_train(self):
        """FleetManager synchronizes and trains."""
        fleet = FleetManager(num_robots=5, num_tasks=10)
        fleet.start_data_collection(duration_days=1)
        fleet.sync_and_train()

        # Check that model was trained
        assert fleet.central_server.current_model_version == 1

    def test_fleet_manager_health_check(self):
        """FleetManager reports fleet health."""
        fleet = FleetManager(num_robots=10, num_tasks=20)
        health = fleet.get_fleet_health()
        assert isinstance(health, dict)
        assert 'healthy_robots' in health
        assert 'fleet_success_rate' in health


class TestModule6LearningConcepts:
    """Validate that code demonstrates key learning concepts."""

    def test_multi_task_policy_demonstrates_transfer_learning(self):
        """Code shows transfer learning: pre-train → fine-tune."""
        # Check that finetune_on_new_task exists and freezes encoder
        policy = MultiTaskPolicy(num_tasks=10)
        source_code = inspect.getsource(
            policy.finetune_on_new_task
        )
        assert 'requires_grad' in source_code
        assert 'freeze' in source_code.lower() or 'param' in source_code

    def test_multi_task_policy_demonstrates_task_conditioning(self):
        """Code shows task embedding for multi-task learning."""
        # Check TaskEmbedding class
        source = inspect.getsource(TaskEmbedding)
        assert 'embedding' in source.lower()

    def test_multi_task_policy_demonstrates_shared_representation(self):
        """Code shows shared encoder + task-specific heads."""
        policy = MultiTaskPolicy(num_tasks=10)
        # Encoder is shared
        assert hasattr(policy, 'encoder')
        # Heads are task-specific
        assert len(policy.task_heads) == 10

    def test_fleet_manager_demonstrates_federated_learning(self):
        """Code shows federated learning: local updates → aggregation."""
        aggregator = FederatedLearningAggregator(num_robots=10)
        source = inspect.getsource(aggregator.aggregate)
        assert 'average' in source.lower() or 'mean' in source.lower()

    def test_fleet_manager_demonstrates_distributed_coordination(self):
        """Code shows coordination between robots and server."""
        fleet = FleetManager(num_robots=5)
        # Fleet has robots AND central server
        assert len(fleet.robots) > 0
        assert fleet.central_server is not None

    def test_fleet_manager_demonstrates_continuous_learning(self):
        """Code shows continuous improvement loop."""
        source = inspect.getsource(FleetManager)
        assert 'collect' in source.lower()
        assert 'train' in source.lower()
        assert 'deploy' in source.lower()


class TestModule6Documentation:
    """Validate code documentation and docstrings."""

    def test_multi_task_policy_has_docstring(self):
        """MultiTaskPolicy class has docstring."""
        assert MultiTaskPolicy.__doc__ is not None
        assert len(MultiTaskPolicy.__doc__) > 100

    def test_multi_task_policy_methods_documented(self):
        """Key methods have docstrings."""
        methods = ['forward', 'train_step', 'finetune_on_new_task', 'evaluate']
        for method_name in methods:
            method = getattr(MultiTaskPolicy, method_name)
            assert method.__doc__ is not None

    def test_fleet_manager_has_docstring(self):
        """FleetManager class has docstring."""
        assert FleetManager.__doc__ is not None
        assert len(FleetManager.__doc__) > 100

    def test_robot_node_has_docstring(self):
        """RobotNode class has docstring."""
        assert RobotNode.__doc__ is not None
        assert len(RobotNode.__doc__) > 50

    def test_central_server_has_docstring(self):
        """CentralServer class has docstring."""
        assert CentralServer.__doc__ is not None
        assert len(CentralServer.__doc__) > 50


class TestModule6Integration:
    """Integration tests combining multiple components."""

    def test_full_training_pipeline(self):
        """Test complete training: collect → upload → train → deploy."""
        # Initialize
        fleet = FleetManager(num_robots=3, num_tasks=5)

        # Collect
        fleet.start_data_collection(duration_days=1)

        # Sync and train
        fleet.sync_and_train()

        # Verify model updated
        initial_version = fleet.central_server.current_model_version
        assert initial_version == 1

        # Do another cycle
        fleet.start_data_collection(duration_days=1)
        fleet.sync_and_train()

        # Verify model incremented
        assert fleet.central_server.current_model_version > initial_version

    def test_multi_week_learning(self):
        """Test multi-week continuous learning loop."""
        fleet = FleetManager(num_robots=3, num_tasks=5)

        # Simulate 2 weeks
        for week in range(2):
            fleet.start_data_collection(duration_days=7)
            fleet.sync_and_train()

        # Should have trained twice
        assert fleet.central_server.current_model_version == 2

    def test_robot_health_monitoring(self):
        """Test robot health monitoring over time."""
        fleet = FleetManager(num_robots=5, num_tasks=10)
        fleet.start_data_collection(duration_days=1)

        # Check health
        health = fleet.get_fleet_health()
        assert 'healthy_robots' in health
        assert 'fleet_success_rate' in health


class TestModule6CodeQuality:
    """Validate code quality and standards."""

    def test_multi_task_policy_file_length(self):
        """Multi-task policy file is substantial."""
        import multi_task_policy
        source = inspect.getsource(multi_task_policy)
        lines = len(source.split('\n'))
        assert lines > 300

    def test_fleet_manager_file_length(self):
        """Fleet manager file is substantial."""
        import fleet_manager
        source = inspect.getsource(fleet_manager)
        lines = len(source.split('\n'))
        assert lines > 400

    def test_multi_task_policy_has_type_hints(self):
        """Multi-task policy uses type hints."""
        source = inspect.getsource(MultiTaskPolicy.forward)
        assert '->' in source  # Return type annotation

    def test_fleet_manager_has_type_hints(self):
        """Fleet manager uses type hints."""
        source = inspect.getsource(RobotNode.run_trial)
        assert '->' in source  # Return type annotation


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
