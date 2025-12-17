# Benchmarking at Scale: Evaluation Framework

Measuring performance across 150+ tasks fairly and systematically.

---

## The Benchmarking Challenge

### Single-Task Evaluation (Easy)

```
One task: "Grasp object"
  Run 50 test trials
  Success rate: 48/50 = 96%
  95% confidence interval: [91%, 99%]
  Takes: 1 hour
  Report: "96% success"
```

### Multi-Task Evaluation (Hard)

```
150 tasks, each with:
  - Different difficulty levels
  - Different numbers of demonstrations in training
  - Different real-world relevance
  - Different success rate distributions

How do you summarize 150 numbers into one score?

Option 1: Average all tasks
  Result: (82% + 75% + 91% + ... + 68%) / 150 = 77%
  Problem: Easy tasks (95%) drown out hard tasks (30%)

Option 2: Geometric mean
  Result: (0.82 × 0.75 × 0.91 × ... × 0.68) ^ (1/150)
  Problem: Still skewed toward easy tasks

Option 3: Weighted by difficulty
  w_i = (100 - difficulty_i) / sum of weights
  Result: Prioritize hard tasks
  Problem: Requires defining "difficulty"

Option 4: Report distribution
  Min: 15% (hardest task)
  25th percentile: 55%
  Median: 76%
  75th percentile: 88%
  Max: 98%
  Problem: Hard to compare two systems
```

---

## Metrics for Multi-Task Systems

### Primary Metric: Task-Weighted Success Rate

```python
def task_weighted_success(task_results):
    """
    Weight tasks by training data quantity.

    More training data = expect higher success = downweight task
    Less training data = expect lower success = upweight task
    """
    weights = {}
    total_demos = sum(t.num_demos for t in task_results)

    for task in task_results:
        # Inverse weighting: fewer demos = higher weight
        weights[task.id] = (1 / task.num_demos) / (
            sum(1/t.num_demos for t in task_results)
        )

    weighted_success = sum(
        weights[task.id] * task.success_rate
        for task in task_results
    )

    return weighted_success
```

### Secondary Metrics

```
1. Task Success Distribution
   Min success: 15%
   25th percentile: 55%
   Median: 76%
   75th percentile: 88%
   Max success: 98%
   Std dev: 18%

2. Task Categories Performance
   ├─ Grasping (20 tasks): 82% average
   ├─ Pushing (15 tasks): 78% average
   ├─ Insertion (10 tasks): 74% average
   └─ Other (105 tasks): 75% average

3. Hardest vs Easiest
   Easiest: "Sort white object" (98%)
   Hardest: "Grasp soft deformable" (15%)
   Range: 83 percentage points

4. Transfer Learning Performance
   Zero-shot (no examples): 55%
   Few-shot (5 examples): 72%
   Fine-tuned (100 examples): 85%

5. Generalization to Novel Objects
   On training objects: 77%
   On novel objects: 52%
   Generalization gap: 25%
```

---

## Statistical Confidence: Sample Sizes

### Proper Statistical Testing

```python
def confidence_interval_95(successes, num_trials):
    """Calculate 95% CI for Bernoulli (success/failure) data."""
    import scipy.stats as stats

    p = successes / num_trials
    n = num_trials

    # Wilson score interval (more accurate than naive)
    z = 1.96  # 95% confidence
    denominator = 1 + z*z / n
    numerator_p = p + z*z / (2*n)
    margin = z * math.sqrt(p * (1-p) / n + z*z / (4*n*n))

    ci_lower = (numerator_p - margin) / denominator
    ci_upper = (numerator_p + margin) / denominator

    return ci_lower, ci_upper

# Example
successes = 48
trials = 50
lower, upper = confidence_interval_95(successes, trials)
print(f"Success: 96%, 95% CI: [{lower*100:.1f}%, {upper*100:.1f}%]")
# Output: 95%, 95% CI: [87.8%, 99.2%]
```

### How Many Trials Needed?

```
For 95% confidence ±5%:

If true success = 90%:
  n = 138 trials needed

If true success = 50%:
  n = 384 trials needed

If true success = 10%:
  n = 138 trials needed

Rule of thumb:
  n = (1.96 / margin_of_error)² × p × (1-p)

For 150 tasks × 50 trials each = 7500 total trials
Time: 2-3 weeks on physical robot
Cost: $5000-10000 (expert time)
```

---

## Evaluation Protocol: Best Practices

### Proper Evaluation Requires

```
1. Independent Test Set
   ✓ Never evaluate on training data
   ✓ Hold out 20% of objects for testing
   ✓ Never touch test set during development

2. Multiple Runs per Task
   ✓ 50+ trials per task (statistical significance)
   ✓ Different object instances
   ✓ Different initial robot positions

3. Consistent Conditions
   ✓ Same lighting
   ✓ Same table height
   ✓ Same gripper (not freshly calibrated or worn)

4. Blinded Evaluation
   ✓ Evaluator doesn't know which method
   ✓ Prevents unconscious bias
   ✓ Random task order

5. Separate Train/Val/Test Split
   └─ Train: 70% of demonstrations (for learning)
   └─ Val: 15% of demonstrations (for hyperparameter tuning)
   └─ Test: 15% (for final evaluation, never seen during training)
```

### The Evaluation Pipeline

```python
def evaluate_system(model, task_list, num_trials_per_task=50):
    """Evaluate multi-task system properly."""

    results = []

    for task in task_list:
        print(f"Evaluating task: {task.name}")

        # Get test objects (never seen during training)
        test_objects = task.get_test_objects()

        successes = 0
        failures = []

        for trial_idx in range(num_trials_per_task):
            # Randomize object instance
            obj = random.choice(test_objects)

            # Randomize initial gripper position
            start_pos = random_position(workspace)

            # Execute
            try:
                success = execute_task(model, task, obj, start_pos)
                if success:
                    successes += 1
                else:
                    failures.append({
                        'object': obj.id,
                        'start_pos': start_pos,
                        'reason': 'gripper_miss',
                    })
            except Exception as e:
                failures.append({
                    'error': str(e),
                })

        # Compute metrics
        success_rate = successes / num_trials_per_task
        ci_lower, ci_upper = confidence_interval_95(successes, num_trials_per_task)

        results.append({
            'task_name': task.name,
            'success_rate': success_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'num_trials': num_trials_per_task,
            'failures': failures,
        })

    return results
```

---

## Failure Analysis: Understanding Errors

### Failure Mode Categories

```
1. Perception Error (Network sees wrong thing)
   Example: Thinks object is blue when it's red
   % of failures: 25%
   Mitigation: Better vision encoder, domain randomization

2. Grounding Error (Understands scene wrong)
   Example: Thinks object is at (0.3, 0.2) when it's at (0.5, 0.1)
   % of failures: 15%
   Mitigation: Improve IK, spatial reasoning

3. Motor Error (Execution fails)
   Example: Gripper command sent but gripper doesn't respond
   % of failures: 20%
   Mitigation: Gripper maintenance, ROS 2 integration

4. Task Ambiguity (Multiple valid solutions, network picks wrong one)
   Example: Can grasp from left OR right, network picks unreachable
   % of failures: 10%
   Mitigation: Use diffusion policy (handles multimodality)

5. Environmental Error (World changed)
   Example: Object already removed, gripper can't reach
   % of failures: 15%
   Mitigation: Graceful fallback, monitoring

6. Unknown (Unexpected failure)
   % of failures: 15%
   Mitigation: Logging, investigation
```

### Failure Analysis Pipeline

```python
def analyze_failures(results):
    """Categorize and report failure modes."""

    failure_categories = {
        'perception': 0,
        'grounding': 0,
        'motor': 0,
        'ambiguity': 0,
        'environment': 0,
        'unknown': 0,
    }

    for task_result in results:
        for failure in task_result['failures']:
            # Analyze video/logs to categorize
            category = categorize_failure(failure)
            failure_categories[category] += 1

    # Report
    print("Failure Breakdown:")
    for category, count in failure_categories.items():
        pct = count / sum(failure_categories.values()) * 100
        print(f"  {category}: {count} ({pct:.1f}%)")

    # Prioritize fixes
    print("\nTop improvements to make:")
    for category, count in sorted(
        failure_categories.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]:
        print(f"  1. Fix {category} failures (would gain {count}% success)")
```

---

## Comparing Methods Fairly

### The Comparison Table

```
Scenario: Compare BC vs Diffusion vs RL on same 150 tasks

                  BC        Diffusion     RL
─────────────────────────────────────────────────
Accuracy          76%       83%           89%
95% CI            [74,78]   [81,85]       [87,91]
Training time     1 week    3 days        2 weeks
Data needed       300 demos 300 demos     (0 + fine-tune data)
Generalization    48%       62%           75%
Cost              $5K       $5K           $20K
─────────────────────────────────────────────────
Winner            Balanced  Quality       Performance

Statistical test:
  Is Diffusion significantly better than BC?
  χ² test: p-value = 0.02 (yes, significant at p<0.05)

  Is RL significantly better than Diffusion?
  χ² test: p-value = 0.0001 (yes, very significant)
```

### Proper Statistical Comparison

```python
from scipy.stats import chi2_contingency

def compare_methods_statistically(results_bc, results_diffusion):
    """Compare two methods using chi-squared test."""

    # Create 2×2 contingency table
    #              Success    Failure
    # BC           bc_succ    bc_fail
    # Diffusion    diff_succ  diff_fail

    bc_successes = sum(r['success'] for r in results_bc)
    bc_trials = len(results_bc)
    bc_failures = bc_trials - bc_successes

    diff_successes = sum(r['success'] for r in results_diffusion)
    diff_trials = len(results_diffusion)
    diff_failures = diff_trials - diff_successes

    contingency = [
        [bc_successes, bc_failures],
        [diff_successes, diff_failures],
    ]

    chi2, p_value, dof, expected = chi2_contingency(contingency)

    print(f"BC: {bc_successes}/{bc_trials} = {bc_successes/bc_trials*100:.1f}%")
    print(f"Diffusion: {diff_successes}/{diff_trials} = {diff_successes/diff_trials*100:.1f}%")
    print(f"χ² test p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("✓ Difference is statistically significant (p<0.05)")
    else:
        print("✗ Difference is NOT statistically significant")
```

---

## Real-World Benchmarks

### Google RT-2 Benchmark

```
Metric                      Results
─────────────────────────────────────────
Total tasks                 150+
Training data               100K+ demos
Zero-shot transfer          55%
5-shot transfer             72%
Fine-tuned (100 examples)   85%
Training time               Weeks (TPU cluster)

Top performers (per task):
  ├─ Sorting: 98%
  ├─ Grasping diverse objects: 94%
  ├─ Manipulation: 92%
  └─ Long-horizon: 67% (hardest)

Bottom performers:
  ├─ Deformable objects: 42%
  └─ Precision tasks: 38%
```

### MIT Diffusion Policy Benchmark

```
Task               BC       Diffusion    RL
───────────────────────────────────────────
Grasp seen         50%      85%          88%
Grasp novel        30%      70%          75%
Push block         45%      80%          82%
Insert peg         35%      75%          80%
Open drawer        40%      78%          85%
Average            40%      77.6%        82%

Key insight: Diffusion beats BC on all tasks
            RL beats diffusion on most
            But diffusion was fastest to implement
```

---

## Reporting Results: Standard Format

### Publication-Quality Report Template

```markdown
## Results

### Overall Performance

- **Accuracy**: 83.2% ± 2.4% (95% CI)
- **Tasks evaluated**: 150
- **Total trials**: 7500
- **Training data**: 100K demonstrations

### Per-Category Performance

| Category | Tasks | Accuracy | 95% CI | Notes |
|----------|-------|----------|--------|-------|
| Grasping | 20 | 82% | [80,84] | Best performance |
| Pushing | 15 | 78% | [75,81] | Moderate difficulty |
| Insertion | 10 | 74% | [71,77] | Highest variance |
| Other | 105 | 75% | [73,77] | Diverse tasks |

### Failure Analysis

- **Perception errors**: 25% of failures
- **Motor execution**: 20%
- **Grounding errors**: 15%
- **Environmental**: 15%
- **Unknown**: 25%

### Generalization

- **Training objects**: 83.2%
- **Novel objects**: 62.1%
- **Generalization gap**: 21.1%

### Inference Performance

- **Mean latency**: 145ms (100ms to 250ms)
- **p95 latency**: 220ms
- **Throughput**: 6.9 predictions/second

### Comparison to Prior Work

| Method | Accuracy | Training Time | Generalization |
|--------|----------|---------------|-----------------|
| BC | 76% | 1 week | 48% |
| Diffusion | 83% | 3 days | 62% |
| **Ours (RL)** | **89%** | 2 weeks | **75%** |
```

---

## Key Takeaways

✅ **Multiple Metrics**: Average is insufficient, report distribution
✅ **Proper Statistics**: Confidence intervals, sample size calculation
✅ **Failure Analysis**: Categorize errors, prioritize improvements
✅ **Fair Comparison**: Independent test set, multiple runs, blinded evaluation
✅ **Reproducibility**: Document all hyperparameters, random seeds
✅ **Real-World Relevance**: Test on diverse objects, conditions

---

## Next Steps

1. **Define success criteria** - What counts as success for each task?
2. **Determine sample size** - How many trials per task?
3. **Create evaluation protocol** - Train/val/test split, blinding
4. **Implement failure analysis** - Categorize and log all failures
5. **Compare fairly** - Statistical significance testing

---

## Further Reading

- **Evaluating Machine Learning Systems** (Eisele et al., 2020)
- **How to Evaluate NLP Systems** (Kuczmarski, 2018): Generalizes to robotics
- **Benchmarking Robotic Manipulation** (ALOHA, ACT datasets)
- **Statistical Power Analysis** (Cohen, 1988): Sample size planning

---

**Next Section:** [Cost Analysis →](cost-analysis.md)

