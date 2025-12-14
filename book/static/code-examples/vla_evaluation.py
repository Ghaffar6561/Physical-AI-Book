"""
VLA System Evaluation Metrics

This module provides tools to evaluate Vision-Language-Action policies.
Implements metrics from the Physical AI book for measuring VLA success,
transfer to novel tasks, and failure mode diagnosis.

Learning Goals:
  - Understand VLA evaluation metrics beyond simple success rate
  - Learn how to measure semantic understanding vs motor control
  - Diagnose where VLA systems fail
  - Compare zero-shot, few-shot, and fine-tuned approaches

Example:
  >>> evaluator = VLAEvaluator()
  >>> evaluator.add_trial(success=True, prediction_confidence=0.95, ...)
  >>> evaluator.print_report()
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class FailureMode(Enum):
    """Categorize VLA failures by root cause."""
    PERCEPTION_FAILURE = "perception"  # LLM couldn't see object
    LANGUAGE_FAILURE = "language"      # LLM misunderstood instruction
    GROUNDING_FAILURE = "grounding"    # Image → 3D coordinate wrong
    ACTION_FAILURE = "action"          # Predicted invalid action (OOW, collision)
    MOTOR_FAILURE = "motor"            # Robot couldn't execute (IK failed, collision)
    GRASP_FAILURE = "grasp"            # Object slipped or dropped
    PLACEMENT_FAILURE = "placement"    # Object unstable at destination
    TIMING_FAILURE = "timing"          # Latency caused failure
    UNKNOWN = "unknown"                # Unknown failure mode


@dataclass
class Trial:
    """Single trial result (sim or real)."""
    success: bool
    task_description: str
    object_class: str  # "cup", "cube", "bottle", etc.
    task_type: str    # "pick-and-place", "stacking", "manipulation", etc.
    failure_mode: Optional[FailureMode] = None

    # Confidence metrics
    perception_confidence: float = 1.0  # How confident was object detection?
    language_confidence: float = 1.0    # How confident in instruction understanding?
    action_confidence: float = 1.0      # How confident in action prediction?

    # Quality metrics
    position_error_m: float = 0.0       # Distance from target (meters)
    orientation_error_deg: float = 0.0  # Orientation error (degrees)
    gripper_force_error_N: float = 0.0  # Force discrepancy (Newtons)
    execution_time_s: float = 0.0       # Time to complete task

    # Transfer metrics
    is_novel_object: bool = False       # Object not seen during training?
    is_novel_instruction: bool = False  # Instruction phrasing not seen before?

    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class VLAEvaluator:
    """Evaluate Vision-Language-Action policies on various metrics."""

    def __init__(self, task_name: str = "manipulation", sample_size: int = 100):
        """
        Initialize evaluator.

        Args:
            task_name: Name of task being evaluated (e.g., "pick-and-place")
            sample_size: Recommended number of trials for statistical validity
        """
        self.task_name = task_name
        self.sample_size = sample_size
        self.trials: List[Trial] = []

    def add_trial(
        self,
        success: bool,
        task_description: str,
        object_class: str,
        task_type: str,
        failure_mode: Optional[str] = None,
        perception_confidence: float = 1.0,
        language_confidence: float = 1.0,
        action_confidence: float = 1.0,
        position_error_m: float = 0.0,
        orientation_error_deg: float = 0.0,
        gripper_force_error_N: float = 0.0,
        execution_time_s: float = 0.0,
        is_novel_object: bool = False,
        is_novel_instruction: bool = False,
    ):
        """
        Record a single trial.

        Args:
            success: Did the task succeed?
            task_description: What was the robot asked to do?
            object_class: Type of object ("cup", "cube", etc.)
            task_type: Category of task ("pick-and-place", "stacking", etc.)
            failure_mode: If failed, what was the root cause?
            perception_confidence: LLM confidence in object detection (0-1)
            language_confidence: LLM confidence in instruction understanding (0-1)
            action_confidence: LLM confidence in action prediction (0-1)
            position_error_m: How far from target (meters)
            orientation_error_deg: Orientation error (degrees)
            gripper_force_error_N: Force error (Newtons)
            execution_time_s: Task duration (seconds)
            is_novel_object: Object not seen during training?
            is_novel_instruction: Instruction not seen during training?
        """
        # Convert failure mode string to enum
        failure_enum = None
        if failure_mode:
            try:
                failure_enum = FailureMode[failure_mode.upper()]
            except KeyError:
                failure_enum = FailureMode.UNKNOWN

        trial = Trial(
            success=success,
            task_description=task_description,
            object_class=object_class,
            task_type=task_type,
            failure_mode=failure_enum,
            perception_confidence=perception_confidence,
            language_confidence=language_confidence,
            action_confidence=action_confidence,
            position_error_m=position_error_m,
            orientation_error_deg=orientation_error_deg,
            gripper_force_error_N=gripper_force_error_N,
            execution_time_s=execution_time_s,
            is_novel_object=is_novel_object,
            is_novel_instruction=is_novel_instruction,
        )
        self.trials.append(trial)

    def success_rate(self, trials: Optional[List[Trial]] = None) -> float:
        """Compute overall success rate (%)."""
        if trials is None:
            trials = self.trials

        if not trials:
            return 0.0

        successes = sum(1 for t in trials if t.success)
        return 100.0 * successes / len(trials)

    def success_by_category(self) -> Dict[str, float]:
        """Success rate broken down by task type, object class, etc."""
        results = {}

        # By task type
        results['by_task_type'] = {}
        for task_type in set(t.task_type for t in self.trials):
            subset = [t for t in self.trials if t.task_type == task_type]
            results['by_task_type'][task_type] = self.success_rate(subset)

        # By object class
        results['by_object_class'] = {}
        for obj_class in set(t.object_class for t in self.trials):
            subset = [t for t in self.trials if t.object_class == obj_class]
            results['by_object_class'][obj_class] = self.success_rate(subset)

        # Novel vs seen objects
        novel_trials = [t for t in self.trials if t.is_novel_object]
        seen_trials = [t for t in self.trials if not t.is_novel_object]
        results['novel_objects'] = self.success_rate(novel_trials) if novel_trials else 0.0
        results['seen_objects'] = self.success_rate(seen_trials) if seen_trials else 0.0

        # Novel vs seen instructions
        novel_instr = [t for t in self.trials if t.is_novel_instruction]
        seen_instr = [t for t in self.trials if not t.is_novel_instruction]
        results['novel_instructions'] = self.success_rate(novel_instr) if novel_instr else 0.0
        results['seen_instructions'] = self.success_rate(seen_instr) if seen_instr else 0.0

        return results

    def failure_analysis(self) -> Dict[str, int]:
        """Count failures by root cause."""
        failures = {}

        for trial in self.trials:
            if not trial.success and trial.failure_mode:
                mode_name = trial.failure_mode.value
                failures[mode_name] = failures.get(mode_name, 0) + 1

        return failures

    def confidence_analysis(self) -> Dict[str, Dict[str, float]]:
        """Analyze LLM confidence metrics."""
        successful = [t for t in self.trials if t.success]
        failed = [t for t in self.trials if not t.success]

        def compute_stats(trials, metric_name):
            if not trials:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            values = [getattr(t, metric_name) for t in trials]
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }

        return {
            'perception_confidence': {
                'successful': compute_stats(successful, 'perception_confidence'),
                'failed': compute_stats(failed, 'perception_confidence'),
            },
            'language_confidence': {
                'successful': compute_stats(successful, 'language_confidence'),
                'failed': compute_stats(failed, 'language_confidence'),
            },
            'action_confidence': {
                'successful': compute_stats(successful, 'action_confidence'),
                'failed': compute_stats(failed, 'action_confidence'),
            },
        }

    def position_error_analysis(self) -> Dict[str, float]:
        """Analyze positioning accuracy."""
        successful = [t for t in self.trials if t.success]
        failed = [t for t in self.trials if not t.success]

        successful_errors = [t.position_error_m for t in successful]
        failed_errors = [t.position_error_m for t in failed]

        return {
            'successful_mean_error_m': float(np.mean(successful_errors)) if successful_errors else 0.0,
            'successful_std_error_m': float(np.std(successful_errors)) if successful_errors else 0.0,
            'failed_mean_error_m': float(np.mean(failed_errors)) if failed_errors else 0.0,
            'failed_std_error_m': float(np.std(failed_errors)) if failed_errors else 0.0,
        }

    def transfer_diagnosis(self) -> str:
        """Diagnostic report identifying transfer gaps."""
        report = []

        report.append("=" * 70)
        report.append("VLA TRANSFER DIAGNOSIS")
        report.append("=" * 70)

        # Overall performance
        overall_sr = self.success_rate()
        report.append(f"\nOverall Success Rate: {overall_sr:.1f}%")

        # Performance breakdown
        by_category = self.success_by_category()

        report.append(f"\nSuccess by Task Type:")
        for task_type, sr in by_category['by_task_type'].items():
            report.append(f"  {task_type:20s}: {sr:5.1f}%")

        report.append(f"\nSuccess by Object Class:")
        for obj_class, sr in by_category['by_object_class'].items():
            report.append(f"  {obj_class:20s}: {sr:5.1f}%")

        # Generalization
        novel_obj_sr = by_category['novel_objects']
        seen_obj_sr = by_category['seen_objects']
        report.append(f"\nGeneralization:")
        report.append(f"  Seen objects:  {seen_obj_sr:.1f}%")
        report.append(f"  Novel objects: {novel_obj_sr:.1f}%")

        if novel_obj_sr > 0 and seen_obj_sr > 0:
            generalization_gap = seen_obj_sr - novel_obj_sr
            report.append(f"  Generalization gap: {generalization_gap:.1f}% (smaller is better)")

        # Failure modes
        failures = self.failure_analysis()
        if failures:
            report.append(f"\nFailure Modes:")
            total_failures = sum(failures.values())
            for mode, count in sorted(failures.items(), key=lambda x: -x[1]):
                percentage = 100.0 * count / total_failures
                report.append(f"  {mode:20s}: {count:3d} ({percentage:5.1f}%)")

            # Specific recommendations
            report.append(f"\nRecommendations:")

            if failures.get('perception', 0) > total_failures * 0.2:
                report.append("  ⚠ Perception failures are high (>20%)")
                report.append("    - Improve object detection accuracy")
                report.append("    - Add more diverse training images")
                report.append("    - Check camera calibration")

            if failures.get('language', 0) > total_failures * 0.2:
                report.append("  ⚠ Language understanding issues")
                report.append("    - Fine-tune LLM on task-specific instructions")
                report.append("    - Add ambiguity resolution (ask clarifying questions)")

            if failures.get('grounding', 0) > total_failures * 0.15:
                report.append("  ⚠ Image-to-3D grounding errors")
                report.append("    - Recalibrate camera intrinsics/extrinsics")
                report.append("    - Verify depth sensor accuracy")

            if failures.get('motor', 0) > total_failures * 0.15:
                report.append("  ⚠ Motor execution issues")
                report.append("    - Check IK solver validity")
                report.append("    - Reduce motion speed for stability")
                report.append("    - Verify joint limit constraints")

        # Confidence analysis
        conf_analysis = self.confidence_analysis()
        report.append(f"\nConfidence Analysis:")
        report.append(f"  Perception confidence:")
        report.append(f"    Successful: {conf_analysis['perception_confidence']['successful']['mean']:.2f}")
        report.append(f"    Failed:     {conf_analysis['perception_confidence']['failed']['mean']:.2f}")
        report.append(f"  Language confidence:")
        report.append(f"    Successful: {conf_analysis['language_confidence']['successful']['mean']:.2f}")
        report.append(f"    Failed:     {conf_analysis['language_confidence']['failed']['mean']:.2f}")
        report.append(f"  Action confidence:")
        report.append(f"    Successful: {conf_analysis['action_confidence']['successful']['mean']:.2f}")
        report.append(f"    Failed:     {conf_analysis['action_confidence']['failed']['mean']:.2f}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)

    def print_report(self):
        """Print comprehensive VLA evaluation report."""
        report = []

        report.append("=" * 70)
        report.append(f"VLA EVALUATION: {self.task_name}")
        report.append("=" * 70)

        # Sample size
        report.append(f"\nSample Size:")
        report.append(f"  Trials: {len(self.trials)}")
        if len(self.trials) < self.sample_size:
            report.append(f"  Recommendation: Collect {self.sample_size - len(self.trials)} more trials")

        # Success rate
        overall_sr = self.success_rate()
        report.append(f"\nSuccess Rate: {overall_sr:.1f}%")

        # Execution time
        times = [t.execution_time_s for t in self.trials if t.execution_time_s > 0]
        if times:
            report.append(f"\nExecution Time:")
            report.append(f"  Mean: {np.mean(times):.2f} s")
            report.append(f"  Std:  {np.std(times):.2f} s")
            report.append(f"  Min:  {np.min(times):.2f} s")
            report.append(f"  Max:  {np.max(times):.2f} s")

        # Position accuracy
        pos_analysis = self.position_error_analysis()
        report.append(f"\nPosition Error (successful grasps):")
        report.append(f"  Mean: {pos_analysis['successful_mean_error_m']:.4f} m")
        report.append(f"  Std:  {pos_analysis['successful_std_error_m']:.4f} m")

        print("\n".join(report))
        print()

        # Print diagnosis
        print(self.transfer_diagnosis())

    def export_results(self, filename: str):
        """Export results to JSON for external analysis."""
        import json

        data = {
            'metadata': {
                'task_name': self.task_name,
                'num_trials': len(self.trials),
                'export_timestamp': datetime.now().isoformat(),
            },
            'metrics': {
                'overall_success_rate': self.success_rate(),
                'success_by_category': self.success_by_category(),
                'failure_analysis': self.failure_analysis(),
                'confidence_analysis': self.confidence_analysis(),
                'position_error_analysis': self.position_error_analysis(),
            },
            'trials': [
                {
                    'success': t.success,
                    'task_description': t.task_description,
                    'object_class': t.object_class,
                    'failure_mode': t.failure_mode.value if t.failure_mode else None,
                    'position_error_m': t.position_error_m,
                    'execution_time_s': t.execution_time_s,
                    'is_novel_object': t.is_novel_object,
                    'timestamp': t.timestamp,
                }
                for t in self.trials
            ],
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results exported to {filename}")


def main():
    """Demonstrate VLA evaluation."""
    print("=" * 70)
    print("VLA System Evaluation Demo")
    print("=" * 70)

    # Create evaluator
    evaluator = VLAEvaluator(task_name="pick_and_place")

    # Add sample trials (simulated results)
    print("\n1. Adding simulated trials...")

    # Scenario 1: Trained objects, seen instructions
    for i in range(20):
        success = np.random.rand() > 0.1  # 90% success
        evaluator.add_trial(
            success=success,
            task_description="Pick up the cube",
            object_class="cube",
            task_type="pick-and-place",
            failure_mode=None if success else "grasp",
            perception_confidence=0.95,
            language_confidence=0.95,
            action_confidence=0.92,
            position_error_m=0.01 if success else 0.05,
            execution_time_s=3.5 + np.random.randn() * 0.5,
            is_novel_object=False,
            is_novel_instruction=False,
        )

    # Scenario 2: Novel objects
    for i in range(15):
        success = np.random.rand() > 0.3  # 70% success
        evaluator.add_trial(
            success=success,
            task_description="Pick up the new object",
            object_class="cylinder" if i < 5 else "sphere",
            task_type="pick-and-place",
            failure_mode=None if success else np.random.choice(['perception', 'grasp', 'motor']),
            perception_confidence=0.75,  # Lower on novel objects
            language_confidence=0.90,
            action_confidence=0.80,
            position_error_m=0.02 if success else 0.08,
            execution_time_s=4.0 + np.random.randn() * 0.8,
            is_novel_object=True,
            is_novel_instruction=False,
        )

    # Scenario 3: Novel instructions
    for i in range(10):
        success = np.random.rand() > 0.4  # 60% success
        evaluator.add_trial(
            success=success,
            task_description=f"Execute novel instruction variation {i}",
            object_class="cube",
            task_type="pick-and-place",
            failure_mode=None if success else "language",
            perception_confidence=0.90,
            language_confidence=0.70,  # Lower on novel instructions
            action_confidence=0.75,
            position_error_m=0.015 if success else 0.10,
            execution_time_s=3.5 + np.random.randn() * 1.0,
            is_novel_object=False,
            is_novel_instruction=True,
        )

    print(f"Added {len(evaluator.trials)} trials")

    # Print report
    print("\n2. VLA Evaluation Report:")
    print("-" * 70)
    evaluator.print_report()

    # Export results
    print("\n3. Exporting results...")
    evaluator.export_results('/tmp/vla_results.json')

    print("\n" + "=" * 70)
    print("Usage in Your VLA System:")
    print("=" * 70)
    print("""
# Track VLA policy performance in real-time

evaluator = VLAEvaluator(task_name="manipulation")

for trial_idx in range(num_trials):
    image = camera.get_frame()
    instruction = user_instruction

    # Infer action
    action, confidence = policy.infer(image, instruction)

    # Execute
    success = execute_action(action)

    # Log trial
    evaluator.add_trial(
        success=success,
        task_description=instruction,
        object_class=detected_object_class,
        task_type="manipulation",
        failure_mode=detect_failure_mode(image),
        perception_confidence=confidence['perception'],
        language_confidence=confidence['language'],
        action_confidence=confidence['action'],
        is_novel_object=is_unseen_object(detected_object_class),
    )

# Generate report
evaluator.print_report()
    """)

    print("=" * 70)


if __name__ == '__main__':
    main()
