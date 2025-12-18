# Vision-Language-Action Systems: End-to-End Pipeline

## Introduction

This module integrates all components of Vision-Language-Action (VLA) systems into a complete end-to-end pipeline. We'll implement a working system that processes visual input, interprets natural language commands, and executes robotic actions. This serves as both a capstone for the VLA systems module and a foundation for the capstone project.

## Complete VLA System Architecture

```python
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
from typing import Dict, List, Any, Tuple
import asyncio
import time

class VisionLanguageActionSystem:
    """
    Complete end-to-end VLA system combining vision, language, and action components.
    """
    def __init__(self, config_file="vla_config.json"):
        """
        Initialize the complete VLA system.
        
        Args:
            config_file: Path to configuration file with model settings
        """
        self.config = self.load_config(config_file)
        
        # Initialize components
        self.vision_processor = VisionProcessor(self.config['vision'])
        self.language_processor = LanguageProcessor(self.config['language'])
        self.multimodal_fusion = MultimodalFusion(
            self.config['fusion']['visual_dim'],
            self.config['fusion']['language_dim'], 
            self.config['fusion']['fusion_dim']
        )
        self.action_generator = ActionGenerator(
            self.config['action']['input_dim'],
            self.config['action']['action_space_dim']
        )
        self.action_executor = ActionExecutor(self.config['execution'])
        
        # Performance monitoring
        self.performance_metrics = {
            'total_processed': 0,
            'average_latency': 0,
            'success_rate': 0
        }
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        # Default configuration
        config = {
            'vision': {
                'model_name': 'resnet50',
                'image_size': [224, 224],
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'language': {
                'model_name': 'distilbert-base-uncased',
                'max_length': 64,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'fusion': {
                'visual_dim': 2048,
                'language_dim': 768,
                'fusion_dim': 512
            },
            'action': {
                'input_dim': 512,
                'action_space_dim': 100,  # Adjust based on robot capabilities
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'execution': {
                'robot_interface': 'simulated',
                'max_action_time': 30.0  # seconds
            }
        }
        
        # Try to load from file, use defaults if file doesn't exist
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except FileNotFoundError:
            print(f"Config file {config_file} not found, using defaults")
        
        return config
    
    def process_command(self, image: Image.Image, command: str) -> Dict[str, Any]:
        """
        Process a vision-language command through the complete pipeline.
        
        Args:
            image: Input image from robot's camera
            command: Natural language command from user
            
        Returns:
            Dictionary containing actions and metadata
        """
        start_time = time.time()
        
        # Step 1: Process visual input
        visual_features = self.vision_processor.process_image(image)
        
        # Step 2: Process language input
        language_features = self.language_processor.encode_command(command)
        
        # Step 3: Fuse modalities
        fused_features = self.multimodal_fusion(
            visual_features, 
            language_features
        )
        
        # Step 4: Generate actions
        raw_actions = self.action_generator(fused_features)
        actions = self.post_process_actions(raw_actions)
        
        # Step 5: Execute actions (in simulation or on real robot)
        execution_result = self.action_executor.execute(actions)
        
        # Update performance metrics
        total_time = time.time() - start_time
        self.update_metrics(total_time, execution_result.get('success', True))
        
        return {
            'actions': actions,
            'execution_result': execution_result,
            'processing_time': total_time,
            'visual_features_shape': tuple(visual_features.shape),
            'language_features_shape': tuple(language_features.shape)
        }
    
    def post_process_actions(self, raw_actions: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Post-process raw action outputs into executable commands.
        
        Args:
            raw_actions: Raw action tensor from generator
            
        Returns:
            List of structured action commands
        """
        # Convert tensor to probabilities and get top-k actions
        action_probs = torch.softmax(raw_actions, dim=-1)
        top_k = min(5, action_probs.shape[-1])  # Get top 5 actions
        top_probs, top_indices = torch.topk(action_probs, top_k, dim=-1)
        
        # Map indices to action names and create structured output
        actions = []
        for i in range(top_k):
            action_idx = top_indices[0][i].item()
            action_prob = top_probs[0][i].item()
            
            action_name = self.action_index_to_name(action_idx)
            action_params = self.extract_action_parameters(action_idx)
            
            actions.append({
                'name': action_name,
                'probability': action_prob,
                'parameters': action_params,
                'index': action_idx
            })
        
        return actions
    
    def action_index_to_name(self, index: int) -> str:
        """
        Map action index to human-readable action name.
        This is a simplified mapping; in practice, this would be learned.
        """
        action_names = [
            'navigate_forward', 'navigate_backward', 'turn_left', 'turn_right',
            'grasp_object', 'release_object', 'lift_object', 'lower_object',
            'open_gripper', 'close_gripper', 'detect_object', 'find_person',
            'approach_object', 'move_away', 'stop_robot', 'look_up',
            'look_down', 'pan_left', 'pan_right', 'wave', 'point'
        ] + [f'custom_action_{i}' for i in range(80)]
        
        if index < len(action_names):
            return action_names[index]
        else:
            return f'unknown_action_{index}'
    
    def extract_action_parameters(self, action_index: int) -> Dict[str, Any]:
        """
        Extract action-specific parameters based on action index.
        """
        params = {}
        
        # Some actions require specific parameters
        if action_index in [0, 1]:  # navigate actions
            params['distance'] = 1.0  # meters
        elif action_index in [2, 3]:  # turn actions
            params['angle'] = 90  # degrees
        elif action_index in [4, 5]:  # grasp/release
            params['object_id'] = 'target_object'
            params['grasp_type'] = 'pinch'
        
        return params
    
    def update_metrics(self, processing_time: float, success: bool):
        """
        Update performance metrics.
        """
        self.performance_metrics['total_processed'] += 1
        
        # Update average latency
        old_avg = self.performance_metrics['average_latency']
        n = self.performance_metrics['total_processed']
        new_avg = ((old_avg * (n - 1)) + processing_time) / n
        self.performance_metrics['average_latency'] = new_avg
        
        # Update success rate
        old_success_rate = self.performance_metrics['success_rate']
        if success:
            new_success_rate = ((old_success_rate * (n - 1)) + 1) / n
        else:
            new_success_rate = (old_success_rate * (n - 1)) / n
        self.performance_metrics['success_rate'] = new_success_rate
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        """
        return self.performance_metrics.copy()
```

## Vision Processing Component

```python
import torchvision.transforms as T
from transformers import AutoImageProcessor
import torch.nn.functional as F

class VisionProcessor:
    """
    Component for processing visual input from robot cameras.
    """
    def __init__(self, config: Dict[str, Any]):
        self.device = config['device']
        self.image_size = tuple(config['image_size'])
        
        # Setup image preprocessing pipeline
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize vision encoder
        from transformers import CLIPVisionModel
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_encoder = self.vision_encoder.to(self.device)
        self.vision_encoder.eval()
    
    def process_image(self, image: Image.Image) -> torch.Tensor:
        """
        Process an input image and extract visual features.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Visual features tensor
        """
        # Preprocess image
        processed_image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features using vision encoder
        with torch.no_grad():
            outputs = self.vision_encoder(processed_image)
            # Use the pooled output as visual representation
            visual_features = outputs.pooler_output
        
        return visual_features
```

## Language Processing Component

```python
from transformers import AutoTokenizer

class LanguageProcessor:
    """
    Component for processing natural language commands.
    """
    def __init__(self, config: Dict[str, Any]):
        self.device = config['device']
        self.max_length = config['max_length']
        
        # Initialize tokenizer and language model
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode_command(self, command: str) -> torch.Tensor:
        """
        Encode a natural language command into features.
        
        Args:
            command: Natural language command string
            
        Returns:
            Language features tensor
        """
        # Tokenize the command
        encoded = self.tokenizer(
            command,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # For a complete implementation, we would pass through BERT/other model
        # For now, return the token embeddings as a proxy
        # In practice, use the actual language model
        from transformers import AutoModel
        model = AutoModel.from_pretrained(self.tokenizer.name_or_path)
        model = model.to(self.device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # Use the [CLS] token representation as sentence embedding
            language_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        return language_features
```

## Multimodal Fusion Component

```python
class MultimodalFusion(nn.Module):
    """
    Fuses visual and language features into a joint representation.
    """
    def __init__(self, visual_dim: int, language_dim: int, fusion_dim: int):
        super().__init__()
        
        # Projection layers to bring different modalities to same dimension
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.language_proj = nn.Linear(language_dim, fusion_dim)
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU()
        )
    
    def forward(self, visual_features: torch.Tensor, language_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse visual and language features.
        
        Args:
            visual_features: Features from vision processing
            language_features: Features from language processing
            
        Returns:
            Fused multimodal representation
        """
        # Project both modalities to same dimension
        proj_visual = F.relu(self.visual_proj(visual_features))
        proj_language = F.relu(self.language_proj(language_features))
        
        # Apply cross-attention
        attended_visual, _ = self.cross_attention(
            proj_visual, proj_language, proj_language
        )
        
        # Concatenate and pass through fusion network
        combined = torch.cat([attended_visual, proj_language], dim=-1)
        fused_features = self.fusion_network(combined)
        
        return fused_features
```

## Action Generation Component

```python
class ActionGenerator(nn.Module):
    """
    Generates robot actions from multimodal features.
    """
    def __init__(self, input_dim: int, action_space_dim: int):
        super().__init__()
        
        self.action_decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_space_dim)
        )
    
    def forward(self, multimodal_features: torch.Tensor) -> torch.Tensor:
        """
        Generate action logits from multimodal features.
        
        Args:
            multimodal_features: Fused visual-language features
            
        Returns:
            Action logits
        """
        action_logits = self.action_decoder(multimodal_features)
        return action_logits
```

## Action Execution Component

```python
class ActionExecutor:
    """
    Executes generated actions on the robot (simulated or real).
    """
    def __init__(self, config: Dict[str, Any]):
        self.interface_type = config['robot_interface']
        self.max_action_time = config['max_action_time']
        
        # Initialize robot interface based on type
        if self.interface_type == 'simulated':
            self.robot_interface = SimulatedRobotInterface()
        elif self.interface_type == 'real':
            self.robot_interface = RealRobotInterface()  # Would implement actual interface
        else:
            raise ValueError(f"Unknown robot interface type: {self.interface_type}")
    
    def execute(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a sequence of actions.
        
        Args:
            actions: List of action dictionaries
            
        Returns:
            Execution result
        """
        results = []
        success = True
        error_msg = ""
        
        for action in actions:
            try:
                # Execute the action
                action_result = self.execute_single_action(action)
                results.append(action_result)
                
                # Check if action succeeded
                if not action_result.get('success', False):
                    success = False
                    error_msg = f"Action {action['name']} failed: {action_result['error']}"
                    break  # Stop execution if action fails
                    
            except Exception as e:
                success = False
                error_msg = f"Exception during action execution: {str(e)}"
                results.append({
                    'action': action['name'],
                    'success': False,
                    'error': str(e)
                })
                break
        
        return {
            'success': success,
            'results': results,
            'error_message': error_msg if not success else None,
            'total_actions_executed': len(results)
        }
    
    def execute_single_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single action on the robot.
        
        Args:
            action: Action dictionary containing name and parameters
            
        Returns:
            Execution result
        """
        try:
            # Execute action through robot interface
            result = self.robot_interface.execute_action(
                action['name'], 
                action['parameters']
            )
            
            return {
                'action': action['name'],
                'parameters': action['parameters'],
                'success': result['success'],
                'execution_time': result.get('execution_time', 0),
                'details': result.get('details', {})
            }
        except Exception as e:
            return {
                'action': action['name'],
                'parameters': action['parameters'],
                'success': False,
                'error': str(e)
            }

class SimulatedRobotInterface:
    """
    Simulated robot interface for testing.
    """
    def __init__(self):
        self.position = [0, 0, 0]
        self.orientation = [0, 0, 0]
        self.gripper_state = 'open'
    
    def execute_action(self, action_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action in simulation.
        """
        start_time = time.time()
        
        if action_name == 'navigate_forward':
            self.position[0] += parameters.get('distance', 1.0)
        elif action_name == 'navigate_backward':
            self.position[0] -= parameters.get('distance', 1.0)
        elif action_name == 'turn_left':
            self.orientation[2] += parameters.get('angle', 90)
        elif action_name == 'turn_right':
            self.orientation[2] -= parameters.get('angle', 90)
        elif action_name == 'grasp_object':
            self.gripper_state = 'closed'
        elif action_name == 'release_object':
            self.gripper_state = 'open'
        # Add more actions as needed
        
        execution_time = time.time() - start_time
        
        return {
            'success': True,
            'execution_time': execution_time,
            'details': {
                'position': self.position.copy(),
                'gripper_state': self.gripper_state
            }
        }

class RealRobotInterface:
    """
    Interface for real robot (placeholder).
    """
    def execute_action(self, action_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action on a real robot.
        This is a placeholder that would connect to actual robot control.
        """
        # In a real implementation, this would send commands to the actual robot
        # For now, we'll simulate with some delay and random success
        time.sleep(0.1)  # Simulate communication delay
        
        import random
        success = random.random() > 0.1  # 90% success rate in simulation
        
        return {
            'success': success,
            'execution_time': 0.1,
            'details': {}
        }
```

## Integration with ROS 2

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from PIL import Image as PILImage

class VLAIntegrationNode(Node):
    """
    ROS 2 node that integrates the complete VLA system.
    """
    def __init__(self):
        super().__init__('vla_integration_node')
        
        # Initialize VLA system
        self.vla_system = VisionLanguageActionSystem()
        self.bridge = CvBridge()
        
        # Initialize state
        self.latest_image = None
        self.pending_command = None
        
        # Create subscribers
        self.image_subscription = self.create_subscription(
            ROSImage,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.command_subscription = self.create_subscription(
            String,
            '/vla_commands',
            self.command_callback,
            10
        )
        
        # Create publishers
        self.action_publisher = self.create_publisher(
            String,
            '/vla_actions',
            10
        )
        
        self.status_publisher = self.create_publisher(
            String,
            '/vla_status',
            10
        )
        
        # Process when both image and command are available
        self.process_timer = self.create_timer(0.1, self.process_if_ready)
        
        self.get_logger().info('VLA Integration Node initialized')
    
    def image_callback(self, msg):
        """
        Handle incoming camera images.
        """
        try:
            # Convert ROS Image to PIL Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = PILImage.fromarray(cv_image)
            self.get_logger().debug('Received new image')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def command_callback(self, msg):
        """
        Handle incoming natural language commands.
        """
        self.pending_command = msg.data
        self.get_logger().info(f'Received command: {msg.data}')
    
    def process_if_ready(self):
        """
        Process command if both image and command are available.
        """
        if self.latest_image is not None and self.pending_command is not None:
            try:
                # Process through VLA system
                result = self.vla_system.process_command(
                    self.latest_image,
                    self.pending_command
                )
                
                # Publish execution results
                status_msg = String()
                status_msg.data = json.dumps({
                    'command': self.pending_command,
                    'actions': result['actions'],
                    'success': result['execution_result']['success'],
                    'processing_time': result['processing_time']
                })
                self.status_publisher.publish(status_msg)
                
                # Publish individual actions
                for action in result['actions']:
                    action_msg = String()
                    action_msg.data = json.dumps(action)
                    self.action_publisher.publish(action_msg)
                
                self.get_logger().info(f'Processed command: {self.pending_command}')
                
                # Clear command after processing
                self.pending_command = None
                
            except Exception as e:
                self.get_logger().error(f'Error processing command: {e}')
    
    def get_performance_report(self):
        """
        Get performance report for the VLA system.
        """
        return self.vla_system.get_performance_report()

def main(args=None):
    """
    Main function to run the VLA Integration Node.
    """
    rclpy.init(args=args)
    
    vla_node = VLAIntegrationNode()
    
    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        pass
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()
```

## Example Usage and Testing

```python
def test_vla_system():
    """
    Test the complete VLA system with sample data.
    """
    print("Testing Vision-Language-Action System")
    print("=" * 50)
    
    # Initialize the system
    vla_system = VisionLanguageActionSystem()
    
    # Sample inputs
    # In a real system, this would be actual image data
    sample_image = Image.new('RGB', (224, 224), color='red')
    sample_command = "Navigate to the red object and pick it up"
    
    print(f"Command: {sample_command}")
    print(f"Processing...")
    
    # Process the command
    result = vla_system.process_command(sample_image, sample_command)
    
    print(f"Generated {len(result['actions'])} actions:")
    for i, action in enumerate(result['actions']):
        print(f"  {i+1}. {action['name']} (confidence: {action['probability']:.3f})")
    
    print(f"Execution result: {result['execution_result']['success']}")
    print(f"Processing time: {result['processing_time']:.3f}s")
    
    # Show performance metrics
    metrics = vla_system.get_performance_report()
    print(f"\nPerformance Metrics:")
    print(f"  Total processed: {metrics['total_processed']}")
    print(f"  Avg latency: {metrics['average_latency']:.3f}s")
    print(f"  Success rate: {metrics['success_rate']:.3f}")
    
    return result

def run_simulation_loop():
    """
    Run a simulation loop with continuous inputs.
    """
    vla_system = VisionLanguageActionSystem()
    
    test_commands = [
        "Go to the blue object",
        "Pick up the small cube",
        "Navigate to the table and wait",
        "Find the person and greet them"
    ]
    
    print("Starting VLA System Simulation Loop...")
    print("Press Ctrl+C to exit")
    
    import time
    import random
    
    try:
        while True:
            # Generate a random test command
            command = random.choice(test_commands)
            image = Image.new('RGB', (224, 224), color='blue')
            
            print(f"\nProcessing: {command}")
            
            result = vla_system.process_command(image, command)
            
            print(f"Result: {result['execution_result']['success']}")
            
            time.sleep(2)  # Wait 2 seconds between commands
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
        
        # Show final metrics
        metrics = vla_system.get_performance_report()
        print(f"\nFinal Performance Metrics:")
        print(f"  Total processed: {metrics['total_processed']}")
        print(f"  Avg latency: {metrics['average_latency']:.3f}s")
        print(f"  Success rate: {metrics['success_rate']:.3f}")

if __name__ == "__main__":
    # Run the test
    test_vla_system()
    
    print("\n" + "="*50)
    print("Running continuous simulation (Press Ctrl+C to exit)")
    run_simulation_loop()
```

## Error Handling and Safety

```python
class SafeVLAWrapper:
    """
    A safety wrapper around the VLA system to ensure safe operation.
    """
    def __init__(self, vla_system):
        self.vla_system = vla_system
        self.safety_constraints = self.define_safety_constraints()
    
    def define_safety_constraints(self):
        """
        Define safety constraints for the VLA system.
        """
        return {
            'max_navigation_distance': 10.0,  # meters
            'max_payload': 5.0,  # kg
            'max_execution_time': 60.0,  # seconds
            'forbidden_actions': ['move_to_unsafe_location', 'grasp_very_hot_object'],
            'required_safety_checks': ['collision_avoidance', 'human_safe_zone']
        }
    
    def safe_process_command(self, image: Image.Image, command: str) -> Dict[str, Any]:
        """
        Process command with safety checks.
        """
        # Pre-processing safety check
        if not self.validate_command(command):
            return {
                'success': False,
                'error': 'Command validation failed',
                'actions': []
            }
        
        try:
            # Process command normally
            result = self.vla_system.process_command(image, command)
            
            # Post-processing safety check
            safe_actions = self.filter_safe_actions(result['actions'])
            
            if len(safe_actions) != len(result['actions']):
                print(f"Filtered {len(result['actions']) - len(safe_actions)} unsafe actions")
            
            # Update result with safe actions
            result['actions'] = safe_actions
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'System error: {str(e)}',
                'actions': []
            }
    
    def validate_command(self, command: str) -> bool:
        """
        Validate command is safe to process.
        """
        command_lower = command.lower()
        
        # Check for forbidden commands
        for forbidden in self.safety_constraints['forbidden_actions']:
            if forbidden in command_lower:
                return False
        
        # Check for other safety concerns
        if 'emergency' in command_lower or 'danger' in command_lower:
            # Handle emergency commands appropriately
            return True  # Could have special handling for emergency commands
        
        return True
    
    def filter_safe_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter actions to remove unsafe ones.
        """
        safe_actions = []
        
        for action in actions:
            if self.is_action_safe(action):
                safe_actions.append(action)
        
        return safe_actions
    
    def is_action_safe(self, action: Dict[str, Any]) -> bool:
        """
        Check if individual action is safe to execute.
        """
        action_name = action['name']
        params = action['parameters']
        
        # Check navigation distance
        if action_name in ['navigate_forward', 'navigate_backward', 'move_to_location']:
            distance = params.get('distance', 0)
            if distance > self.safety_constraints['max_navigation_distance']:
                return False
        
        # Check payload
        if action_name == 'grasp_object':
            payload = params.get('weight', 0)
            if payload > self.safety_constraints['max_payload']:
                return False
        
        # Check for forbidden actions
        if action_name in self.safety_constraints['forbidden_actions']:
            return False
        
        return True

def demonstrate_safe_vla():
    """
    Demonstrate the safe VLA system wrapper.
    """
    print("Demonstrating Safe VLA System")
    print("="*40)
    
    # Create base system and wrap with safety
    base_vla = VisionLanguageActionSystem()
    safe_vla = SafeVLAWrapper(base_vla)
    
    # Test with potentially unsafe command
    unsafe_command = "Move to the dangerous area and grasp the hot object"
    image = Image.new('RGB', (224, 224), color='white')
    
    print(f"Processing unsafe command: {unsafe_command}")
    result = safe_vla.safe_process_command(image, unsafe_command)
    
    print(f"Command processed safely: {result.get('success', False)}")
    print(f"Generated {len(result['actions'])} safe actions")
    
    # Test with safe command
    safe_command = "Navigate to the safe location and wait"
    result = safe_vla.safe_process_command(image, safe_command)
    
    print(f"\nProcessing safe command: {safe_command}")
    print(f"Command processed successfully: {result.get('success', False)}")
    print(f"Generated {len(result['actions'])} actions")

if __name__ == "__main__":
    demonstrate_safe_vla()
```

## Conclusion

This end-to-end VLA pipeline demonstrates:

1. **Complete Integration**: All components work together from visual input to action execution
2. **Modular Design**: Each component can be updated independently
3. **Performance Monitoring**: Built-in metrics for system evaluation
4. **Safety Integration**: Safety checks throughout the pipeline
5. **ROS 2 Compatibility**: Ready for integration with robotic systems

The complete VLA system provides a foundation for building sophisticated, language-guided robotic applications. By combining vision, language understanding, and action execution in a unified framework, robots can interact more naturally with humans and adapt to diverse real-world scenarios.