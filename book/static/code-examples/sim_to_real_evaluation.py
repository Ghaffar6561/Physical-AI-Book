"""
Sim-to-Real Transfer Evaluation Metrics.

This module provides tools to measure and diagnose sim-to-real transfer success.
Implements metrics from the Physical AI book for evaluating robot policies.

Learning Goals:
  - Understand transfer success metrics beyond "% success"
  - Learn how to diagnose which sim-to-real gap is causing failures
  - Practice analyzing real robot performance data
  - Compare sim vs real performance quantitatively

Example:
  >>> evaluator = SimToRealEvaluator()
  >>> evaluator.add_sim_trial(success=True, grasp_force=50.0, duration=2.5)
  >>> evaluator.add_real_trial(success=False, gripper_slip=True, reason="wet_bottle")
  >>> evaluator.print_report()
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trial:
    """Single trial result (sim or real)."""
    success: bool
    duration: float  # seconds
    grasp_force: float  # Newtons (for grasping tasks)
    gripper_slip: bool = False
    reason: Optional[str] = None  # Why failure occurred
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class SimToRealEvaluator:
    """Evaluate sim-to-real transfer success and diagnose failures."""

    def __init__(self, task_name: str = "grasping", sample_size: int = 100):
        """
        Initialize evaluator.

        Args:
            task_name: Name of task (e.g., "grasping", "manipulation")
            sample_size: Recommended sample size for statistics (usually 100+ trials)
        """
        self.task_name = task_name
        self.sample_size = sample_size

        self.sim_trials: List[Trial] = []
        self.real_trials: List[Trial] = []

    def add_sim_trial(
        self,
        success: bool,
        duration: float = 2.5,
        grasp_force: float = 50.0,
        reason: Optional[str] = None
    ):
        """
        Record a simulation trial.

        Args:
            success: Whether task succeeded
            duration: Episode duration (seconds)
            grasp_force: Gripper force applied (Newtons)
            reason: If failed, reason for failure
        """
        trial = Trial(
            success=success,
            duration=duration,
            grasp_force=grasp_force,
            reason=reason
        )
        self.sim_trials.append(trial)

    def add_real_trial(
        self,
        success: bool,
        duration: float = 2.5,
        grasp_force: float = 50.0,
        gripper_slip: bool = False,
        reason: Optional[str] = None
    ):
        """
        Record a real robot trial.

        Args:
            success: Whether task succeeded
            duration: Trial duration (seconds)
            grasp_force: Gripper force applied (Newtons)
            gripper_slip: Whether gripper slipped during grasp
            reason: If failed, failure mode (perception, physics, timing)
        """
        trial = Trial(
            success=success,
            duration=duration,
            grasp_force=grasp_force,
            gripper_slip=gripper_slip,
            reason=reason
        )
        self.real_trials.append(trial)

    def success_rate(self, trials: List[Trial]) -> float:
        """Compute success rate (%)."""
        if not trials:
            return 0.0
        return 100.0 * np.mean([t.success for t in trials])

    def success_ci(self, trials: List[Trial]) -> tuple:
        """
        Compute 95% confidence interval for success rate using Wilson score.

        Args:
            trials: List of trials

        Returns:
            (lower, upper) bounds for 95% CI
        """
        if not trials:
            return (0.0, 0.0)

        n = len(trials)
        successes = sum(1 for t in trials if t.success)
        p = successes / n if n > 0 else 0.5

        # Wilson score interval (better than normal approximation)
        z = 1.96  # 95% confidence
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p * (1-p) / n + z**2 / (4*n**2)) / denominator

        lower = max(0, 100 * (center - margin))
        upper = min(100, 100 * (center + margin))

        return (lower, upper)

    def transfer_ratio(self) -> float:
        """
        Compute transfer ratio: Real success / Sim success.

        A ratio ≥ 0.8 indicates good transfer (goal).
        A ratio < 0.5 indicates poor transfer (need to improve sim).
        """
        sim_sr = self.success_rate(self.sim_trials)
        real_sr = self.success_rate(self.real_trials)

        if sim_sr == 0:
            return 0.0
        return real_sr / sim_sr

    def failure_analysis(self) -> Dict[str, int]:
        """
        Analyze failure modes in real trials.

        Returns:
            Dictionary mapping failure reason → count
        """
        failures = {}
        for trial in self.real_trials:
            if not trial.success and trial.reason:
                failures[trial.reason] = failures.get(trial.reason, 0) + 1

        return failures

    def grasp_force_analysis(self) -> Dict:
        """
        Analyze gripper force: successful vs failed grasps.

        Returns:
            Statistics on force usage
        """
        successful_forces = [t.grasp_force for t in self.real_trials if t.success]
        failed_forces = [t.grasp_force for t in self.real_trials if not t.success]

        return {
            'successful': {
                'mean': float(np.mean(successful_forces)) if successful_forces else 0,
                'std': float(np.std(successful_forces)) if successful_forces else 0,
                'min': float(np.min(successful_forces)) if successful_forces else 0,
                'max': float(np.max(successful_forces)) if successful_forces else 0,
                'count': len(successful_forces),
            },
            'failed': {
                'mean': float(np.mean(failed_forces)) if failed_forces else 0,
                'std': float(np.std(failed_forces)) if failed_forces else 0,
                'min': float(np.min(failed_forces)) if failed_forces else 0,
                'max': float(np.max(failed_forces)) if failed_forces else 0,
                'count': len(failed_forces),
            }
        }

    def gripper_slip_analysis(self) -> Dict:
        """
        Analyze gripper slip incidents.

        Returns:
            Slip statistics
        """
        slip_count = sum(1 for t in self.real_trials if t.gripper_slip)
        total_trials = len(self.real_trials)

        return {
            'slip_rate': 100.0 * slip_count / total_trials if total_trials > 0 else 0,
            'slip_count': slip_count,
            'total_trials': total_trials,
        }

    def transfer_diagnosis(self) -> str:
        """
        Diagnose transfer gaps based on failure patterns.

        Returns:
            Diagnostic report with recommendations
        """
        diagnosis = []

        sim_sr = self.success_rate(self.sim_trials)
        real_sr = self.success_rate(self.real_trials)
        ratio = self.transfer_ratio()

        diagnosis.append("=" * 70)
        diagnosis.append("SIM-TO-REAL TRANSFER DIAGNOSIS")
        diagnosis.append("=" * 70)

        # Overall assessment
        diagnosis.append(f"\nTransfer Ratio: {ratio:.2f} ({ratio*100:.1f}%)")
        if ratio >= 0.80:
            diagnosis.append("  Status: GOOD ✓ (acceptable for deployment)")
        elif ratio >= 0.60:
            diagnosis.append("  Status: MODERATE (more work needed)")
        else:
            diagnosis.append("  Status: POOR ✗ (significant gap)")

        # Failure mode analysis
        failures = self.failure_analysis()
        if failures:
            diagnosis.append("\nFailure Modes (Real Robot):")
            for mode, count in sorted(failures.items(), key=lambda x: -x[1]):
                percentage = 100.0 * count / len(self.real_trials)
                diagnosis.append(f"  {mode:25s}: {count:3d} ({percentage:5.1f}%)")

            # Diagnostic recommendations
            diagnosis.append("\nDiagnostic Recommendations:")
            if 'perception' in failures:
                diagnosis.append("  ⚠ Vision failure detected:")
                diagnosis.append("    - Check CNN accuracy on real camera images")
                diagnosis.append("    - Add more lighting randomization to sim")
                diagnosis.append("    - Collect real images for fine-tuning")

            if 'physics' in failures or 'slip' in failures:
                diagnosis.append("  ⚠ Physics failure detected:")
                diagnosis.append("    - Measure real gripper force response")
                diagnosis.append("    - Add motor dynamics to simulation")
                diagnosis.append("    - Randomize friction more aggressively")

            if 'timing' in failures:
                diagnosis.append("  ⚠ Timing failure detected:")
                diagnosis.append("    - Measure actual motor latency")
                diagnosis.append("    - Add realistic command delays to sim")
                diagnosis.append("    - Test with variable message rates")

        # Gripper force analysis
        force_stats = self.grasp_force_analysis()
        if force_stats['failed']['count'] > 0:
            diagnosis.append("\nGripper Force Analysis:")
            diagnosis.append(f"  Successful grasps: {force_stats['successful']['mean']:.1f}±"
                           f"{force_stats['successful']['std']:.1f} N (n={force_stats['successful']['count']})")
            diagnosis.append(f"  Failed grasps:     {force_stats['failed']['mean']:.1f}±"
                           f"{force_stats['failed']['std']:.1f} N (n={force_stats['failed']['count']})")

            if force_stats['failed']['mean'] > force_stats['successful']['mean']:
                diagnosis.append("  → Force policy is too conservative; increase force")
            else:
                diagnosis.append("  → Force policy may be too aggressive; decrease force")

        # Slip analysis
        slip_stats = self.gripper_slip_analysis()
        if slip_stats['slip_rate'] > 5.0:
            diagnosis.append(f"\nGripper Slip Analysis:")
            diagnosis.append(f"  Slip rate: {slip_stats['slip_rate']:.1f}% (n={slip_stats['slip_count']} slips)")
            diagnosis.append("  → Increase gripper friction randomization")
            diagnosis.append("  → Consider tactile feedback in real system")

        return "\n".join(diagnosis)

    def print_report(self):
        """Print comprehensive evaluation report."""
        report = []

        report.append("=" * 70)
        report.append(f"SIM-TO-REAL EVALUATION: {self.task_name}")
        report.append("=" * 70)

        # Sample sizes
        report.append(f"\nSample Sizes:")
        report.append(f"  Simulation trials: {len(self.sim_trials)}")
        report.append(f"  Real robot trials: {len(self.real_trials)}")
        if len(self.real_trials) < self.sample_size:
            report.append(f"  Recommendation: Collect {self.sample_size - len(self.real_trials)} more real trials")

        # Success rates
        sim_sr = self.success_rate(self.sim_trials)
        real_sr = self.success_rate(self.real_trials)

        report.append(f"\nSuccess Rates:")
        report.append(f"  Simulation: {sim_sr:.1f}%")
        if self.sim_trials:
            sim_ci = self.success_ci(self.sim_trials)
            report.append(f"    95% CI: [{sim_ci[0]:.1f}%, {sim_ci[1]:.1f}%]")

        report.append(f"  Real Robot: {real_sr:.1f}%")
        if self.real_trials:
            real_ci = self.success_ci(self.real_trials)
            report.append(f"    95% CI: [{real_ci[0]:.1f}%, {real_ci[1]:.1f}%]")

        # Transfer ratio
        ratio = self.transfer_ratio()
        report.append(f"\nTransfer Ratio: {ratio:.2f}")
        if ratio >= 0.80:
            report.append("  Status: GOOD ✓ (real success ≥ 80% of sim success)")
        elif ratio >= 0.50:
            report.append("  Status: MODERATE (needs improvement)")
        else:
            report.append("  Status: POOR (major gap between sim and real)")

        # Duration analysis
        if self.real_trials:
            real_durations = [t.duration for t in self.real_trials]
            report.append(f"\nTrial Duration (Real):")
            report.append(f"  Mean: {np.mean(real_durations):.2f} s")
            report.append(f"  Std:  {np.std(real_durations):.2f} s")
            report.append(f"  Min:  {np.min(real_durations):.2f} s")
            report.append(f"  Max:  {np.max(real_durations):.2f} s")

        # Print report
        print("\n".join(report))
        print()

        # Print diagnosis
        print(self.transfer_diagnosis())

    def export_results(self, filename: str):
        """
        Export results to CSV for external analysis.

        Args:
            filename: Output CSV file path
        """
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'source', 'success', 'duration', 'grasp_force', 'gripper_slip', 'reason', 'timestamp'
            ])
            writer.writeheader()

            for trial in self.sim_trials:
                writer.writerow({
                    'source': 'sim',
                    'success': trial.success,
                    'duration': trial.duration,
                    'grasp_force': trial.grasp_force,
                    'gripper_slip': trial.gripper_slip,
                    'reason': trial.reason,
                    'timestamp': trial.timestamp,
                })

            for trial in self.real_trials:
                writer.writerow({
                    'source': 'real',
                    'success': trial.success,
                    'duration': trial.duration,
                    'grasp_force': trial.grasp_force,
                    'gripper_slip': trial.gripper_slip,
                    'reason': trial.reason,
                    'timestamp': trial.timestamp,
                })

        print(f"Results exported to {filename}")


def main():
    """Demonstrate sim-to-real evaluation."""
    print("=" * 70)
    print("Sim-to-Real Transfer Evaluation Demo")
    print("=" * 70)

    # Create evaluator
    evaluator = SimToRealEvaluator(task_name="robotic_grasping")

    # Simulate training in sim (perfect conditions)
    print("\n1. Adding simulation trials (100 episodes)...")
    for i in range(100):
        # Simulate: 95% success in sim with domain randomization
        success = np.random.random() < 0.95
        force = np.random.normal(50, 5)  # ~50N ±5N
        evaluator.add_sim_trial(success=success, grasp_force=force)

    # Simulate real deployment (with gaps)
    print("2. Adding real robot trials (50 deployments)...")
    for i in range(50):
        # Real robot: 75% success (sim-to-real gap)
        success = np.random.random() < 0.75

        if not success:
            # Randomly select failure mode
            modes = ['perception', 'physics', 'timing']
            mode = np.random.choice(modes)
            reason = f"{mode}_failure"
            slip = mode == 'physics'
        else:
            reason = None
            slip = False

        force = np.random.normal(48, 8)  # Real robot uses ~48N ±8N (more variation)
        evaluator.add_real_trial(
            success=success,
            grasp_force=force,
            gripper_slip=slip,
            reason=reason
        )

    # Print report
    print("\n3. Evaluation Report:")
    print("-" * 70)
    evaluator.print_report()

    # Export results
    evaluator.export_results('/tmp/sim_to_real_results.csv')

    print("\n" + "=" * 70)
    print("Usage in Your Research:")
    print("=" * 70)
    print("""
evaluator = SimToRealEvaluator(task_name="grasping")

# After sim training:
for sim_trial in sim_trials:
    evaluator.add_sim_trial(success=sim_trial.success, grasp_force=sim_trial.force)

# After real robot testing:
for real_trial in real_trials:
    evaluator.add_real_trial(
        success=real_trial.success,
        grasp_force=real_trial.force,
        reason=real_trial.failure_mode
    )

# Analyze
evaluator.print_report()
ratio = evaluator.transfer_ratio()
if ratio >= 0.8:
    print("Transfer successful! Ready for deployment.")
else:
    print("Transfer gaps remain. Improve sim and retry.")
    """)


if __name__ == '__main__':
    main()
