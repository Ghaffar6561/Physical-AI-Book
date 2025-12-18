# Section 2: Deployment Strategies

Safely deploying models to 100+ robots with zero downtime.

---

## The Deployment Problem

### The Cost of Failure

```
You trained model v24 on latest data.
It's 0.2% better than v23 in lab tests.
Time to deploy to production: ???

Option A: Deploy to all 100 robots immediately
  ✓ Fast (1 minute)
  ✗ High risk: If wrong, 100 robots fail
  ✗ Lost: 1000+ tasks that day (-$50K revenue)
  ✗ Reputation: Customer complaints
  ✗ Recovery: 2 hours to rollback and rebuild trust

Option B: Smart, gradual deployment
  ✓ Low risk: Only 5 robots affected if wrong
  ✓ Validate: Real-world performance before full rollout
  ✓ Recovery: 2 minutes to rollback
  ✗ Slower (20 minutes to full deployment)
  ✓ Confident: High probability of success
```

**This section is about Option B.**

---

## Deployment Strategy 1: Canary Deployment

### What It Is

Deploy to small percentage first, monitor, expand gradually.

```
Timeline:
  T+0min: Deploy v24 to 5% of robots (5 robots)
          Run 100 tasks to validate
  T+5min: Check success rate
          v23: 92% baseline
          v24: 91.8% (within margin, no regression)
  T+10min: Expand to 20% (20 robots)
          Run 500 tasks
  T+20min: Expand to 50% (50 robots)
          Run 2500 tasks
  T+30min: Full rollout to 100% (100 robots)
          Monitor for anomalies
  T+60min: Mark deployment as successful
```

### Success Criteria

```
At each stage, check:
  ✓ Success rate ≥ baseline - 2%  (not worse than v23)
  ✓ No catastrophic failures      (no robot stuck >10x)
  ✓ Latency acceptable            (<200ms p95)
  ✓ Error rate acceptable         (<1% unexpected errors)

If ANY check fails:
  ✗ Immediately rollback to v23
  ✗ Investigate why v24 failed
  ✗ Try again tomorrow
```

### Implementation

```python
class CanaryDeployment:
    def __init__(self, model_v24, baseline_v23):
        self.stages = [
            (0.05, 100),    # 5% of robots, 100 tasks
            (0.20, 500),    # 20% of robots, 500 tasks
            (0.50, 2500),   # 50% of robots, 2500 tasks
            (1.00, 5000),   # 100% of robots, 5000 tasks
        ]
        self.baseline_success = 0.92
        self.tolerance = 0.02  # Allow 2% regression

    def deploy(self):
        for pct, num_tasks in self.stages:
            # Deploy to percentage
            robots = select_random_subset(pct)
            for robot in robots:
                robot.load_model(model_v24)

            # Monitor
            results = monitor_tasks(num_tasks, duration=10_minutes)
            success_rate = results.success_rate

            # Evaluate
            if success_rate < self.baseline_success - self.tolerance:
                # Failure!
                self.rollback()
                self.alert_team("Canary deployment failed")
                return False

            print(f"Stage passed: {pct*100}% at {success_rate*100:.1f}%")

        return True  # Success!

    def rollback(self):
        for robot in all_robots():
            robot.load_model(baseline_v23)
        print("Rolled back to v23")
```

### Pros & Cons

```
✓ Low risk: Failures affect only 5% of fleet initially
✓ Real-world validation: Test on actual data, not lab data
✓ Gradual rollout: Give system time to stabilize
✓ Easy rollback: If something breaks, revert instantly

✗ Slower: Takes 30 minutes for full rollout
✗ Complex monitoring: Need to watch multiple stages
✗ Automation required: Can't do manually for each robot
```

### When to Use

- **Best for**: Frequent deployments (weekly model updates)
- **Best for**: High-confidence models (small changes from v23)
- **Avoid for**: Major architecture changes (use blue-green instead)

---

## Deployment Strategy 2: Blue-Green Deployment

### What It Is

Two complete production environments. Switch traffic between them.

```
BLUE (Current Production):
  ├─ 100 robots running v23
  ├─ Actively handling tasks
  └─ Success rate: 92%

GREEN (New Staging):
  ├─ 100 robots running v24
  ├─ Running shadows/dummy tasks
  ├─ Not affecting customers
  └─ Success rate: ?

Validation (1 day):
  Run 1000 shadow tasks on GREEN
  Compare success to BLUE
  If GREEN is not worse → SWAP

SWAP (instant):
  BLUE (v23) → BLUE_backup
  GREEN (v24) → BLUE_active
  All traffic flows to v24

Result:
  ✓ Zero downtime
  ✓ Instant rollback (reverse swap)
  ✓ High confidence (full-scale validation)
```

### Implementation

```
CURRENT STATE (Day N):
  Load Balancer → BLUE (v23): 100% traffic

NEW VERSION READY (Day N):
  Load Balancer → BLUE (v23): 100% traffic
                → GREEN (v24): 0% traffic, shadow only

VALIDATION RUNNING (1 day):
  GREEN running shadow of 5000 tasks
  Success rate: 91.9% (within bounds)

SWAP DECISION (Day N+1):
  Load Balancer → BLUE (v24): 100% traffic ← SWAPPED!
                → GREEN (v23): 0% traffic, backup

MONITORING (Day N+1 + 4 hours):
  Check v24 success rate: 92.1% ✓
  Confirm no regressions
  Decommission GREEN, keep as backup

CLEANUP (after 1 week):
  Mark v23 as retired
  Archive logs and metrics
  Delete v23 from all robots except backups
```

### Pros & Cons

```
✓ Zero downtime: Instant switch, no customer impact
✓ Easy rollback: One command reverses everything
✓ Full-scale test: Validate on 100% of fleet
✓ High confidence: Can't go wrong once validated

✗ Resource intensive: Need 2× the hardware temporarily
✗ Slower validation: Takes 1+ day to validate
✗ Storage overhead: Keep both versions on disk
```

### When to Use

- **Best for**: Major releases (new architecture)
- **Best for**: Risky changes (major retraining)
- **Best for**: Weekend deployments (have time to monitor)
- **Avoid for**: Frequent small updates (overkill)

---

## Deployment Strategy 3: Shadow Deployment

### What It Is

Run new model in parallel, don't use its decisions, just observe.

```
SHADOW MODE:

Robot gets task:
  ├─ Run v23 → Get action A23
  ├─ Run v24 → Get action A24 (in parallel, shadow)
  ├─ EXECUTE A23 (use current model)
  └─ LOG: Both A23 and A24 for comparison

Later analysis:
  "If we had used v24, would it have succeeded?"
  ├─ A24 == A23? (0% difference in this case)
  ├─ A24 ≠ A23? (Different decision)
  │  └─ A24 would have succeeded? (yes/no)
  └─ Aggregate: 1000 tasks
     └─ v24 would have succeeded 920 times (92% estimated)

Decision:
  If v24 would have done better: Deploy it
  If v24 would have done worse: Retrain
```

### Implementation

```python
def execute_with_shadow():
    """Execute task with both v23 and v24, log results."""

    while task_available():
        task = get_next_task()

        # Get decisions from both models
        action_v23 = model_v23.predict(task.observation)
        action_v24 = model_v24.predict(task.observation)

        # Execute v23 (production)
        result_v23 = robot.execute(action_v23)
        success_v23 = result_v23.success

        # Shadow v24: What would have happened?
        predicted_success_v24 = estimate_success(action_v24, task)

        # Log both
        log_entry = {
            'task_id': task.id,
            'action_v23': action_v23,
            'success_v23': success_v23,
            'action_v24': action_v24,
            'predicted_success_v24': predicted_success_v24,
            'agreement': (action_v23 ≈ action_v24),
        }

        return log_entry

# After 1000 tasks:
results = analyze_shadow_logs()
v24_estimated_success = results.predicted_success_v24.mean()
# If 92%+ → Deploy v24
# If <92% → Keep v23
```

### Pros & Cons

```
✓ Zero risk: Production uses v23, never v24
✓ Real-world data: Test on actual customer tasks
✓ Easy analysis: Log everything, analyze later
✓ Cheap: No extra robots needed

✗ Delayed decision: Takes hours/days to analyze
✗ Estimation error: Prediction != actual execution
✗ Computational overhead: 2× inference on every task
```

### When to Use

- **Best for**: High-risk changes (new architecture)
- **Best for**: Estimating potential improvement
- **Avoid for**: Latency-critical systems (2× inference time)

---

## Deployment Strategy 4: Gradual Rollout (Feature Flags)

### What It Is

Enable new model gradually via runtime feature flag, not version update.

```
VERSION UPDATE: Robots have v24 installed

But:
  ├─ 0%: Using v24 for tasks (feature flag disabled)
  └─ 100%: Using v23 for tasks

Gradually increase:
  T+0min:   enable_v24_flag = 0.0  (0% of tasks)
  T+10min:  enable_v24_flag = 0.1  (10% of tasks)
  T+30min:  enable_v24_flag = 0.5  (50% of tasks)
  T+60min:  enable_v24_flag = 1.0  (100% of tasks)

At each step, monitor success rate.
If regression detected, reduce flag value.
```

### Implementation

```python
def decide_which_model(task):
    """Decide whether to use v23 or v24."""

    # Get feature flag value (0.0 to 1.0)
    v24_probability = config.get_feature_flag('enable_v24')

    # Random decision
    if random.random() < v24_probability:
        return model_v24
    else:
        return model_v23

# Monitoring loop (every 5 minutes)
while True:
    success_v23 = measure_success(model='v23')
    success_v24 = measure_success(model='v24')

    if success_v24 >= success_v23 - 0.02:
        # v24 is good, increase usage
        current_flag = config.get_feature_flag('enable_v24')
        config.set_feature_flag('enable_v24', min(current_flag + 0.1, 1.0))
    else:
        # v24 is bad, decrease usage
        current_flag = config.get_feature_flag('enable_v24')
        config.set_feature_flag('enable_v24', max(current_flag - 0.1, 0.0))

    time.sleep(5 * 60)  # Check every 5 minutes
```

### Pros & Cons

```
✓ Fine-grained control: Adjust rollout in real-time
✓ Fast rollback: Flip flag, instantly back to v23
✓ No redeployment: Code deployed once, behavior changes via config
✓ Works with canary: Combine for even more control

✗ Runtime overhead: Branch on every decision
✗ Monitoring required: Need automated alerts
✗ Complexity: Harder to debug (which model executed?)
```

### When to Use

- **Best for**: Rapid iteration (multiple deployments per day)
- **Best for**: A/B testing (need per-robot bucketing)
- **Combine with**: Canary deployment for safety

---

## A/B Testing in Production

### The Goal

Compare two models on real data with statistical rigor.

```
HYPOTHESIS: Model v24 is better than v23

EXPERIMENT:
  Control: 50 robots use v23
  Test:    50 robots use v24
  Duration: 7 days
  N trials: 10,000 per group

RESULTS:
  Control (v23): 9200/10000 = 92.0% success
  Test (v24):    9210/10000 = 92.1% success
  Difference:    +0.1% (positive but small)

STATISTICAL SIGNIFICANCE:
  Chi-square test: p-value = 0.45 (NOT significant)
  Conclusion: 0.1% gain is likely due to chance
  Decision: Don't deploy (keep v23)
```

### Proper A/B Testing

```
Setup:
  ✓ Random assignment: Each robot randomly gets v23 or v24
  ✓ Blinded: Robots don't know which version
  ✓ Independent: One robot's result doesn't affect another
  ✓ Sufficient sample: 10,000+ trials per group

Monitoring:
  ✓ Track metric continuously: success_rate_per_hour
  ✓ Alert if one variant much worse (stop experiment)
  ✓ Don't peek too early (introduces bias)

Analysis:
  ✓ Chi-square test: p < 0.05 for significance
  ✓ Confidence interval: 95% CI on difference
  ✓ Effect size: Is difference practically meaningful?
  ✓ One-sided or two-sided: What's the hypothesis?
```

### Implementation

```python
class ABTest:
    def __init__(self, model_a, model_b, sample_size=10000):
        self.model_a = model_a
        self.model_b = model_b
        self.sample_size = sample_size
        self.results_a = {'success': 0, 'total': 0}
        self.results_b = {'success': 0, 'total': 0}

    def decide_model(self, robot_id):
        """Assign robot to control or test group."""
        # Deterministic: same robot always gets same model
        hash_val = hash(robot_id) % 2
        return self.model_a if hash_val == 0 else self.model_b

    def log_result(self, robot_id, success):
        """Log trial result."""
        model = self.decide_model(robot_id)
        if model == self.model_a:
            self.results_a['total'] += 1
            if success:
                self.results_a['success'] += 1
        else:
            self.results_b['total'] += 1
            if success:
                self.results_b['success'] += 1

    def analyze(self):
        """Statistical analysis."""
        from scipy.stats import chi2_contingency

        # Contingency table
        contingency = [
            [self.results_a['success'],
             self.results_a['total'] - self.results_a['success']],
            [self.results_b['success'],
             self.results_b['total'] - self.results_b['success']],
        ]

        chi2, p_value, dof, expected = chi2_contingency(contingency)

        success_rate_a = self.results_a['success'] / self.results_a['total']
        success_rate_b = self.results_b['success'] / self.results_b['total']

        return {
            'success_a': success_rate_a,
            'success_b': success_rate_b,
            'difference': success_rate_b - success_rate_a,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }
```

---

## Rollback Procedures

### Fast Rollback

If something goes wrong, revert instantly:

```
ROLLBACK v24 → v23 (2 minutes):

Step 1: Stop v24 (5 sec)
  flag.disable_v24()  # Stop using v24

Step 2: Load v23 (30 sec)
  all_robots.load_model(v23)

Step 3: Verify (30 sec)
  check_success_rate() >= 90%

Step 4: Alert (1 min)
  notify_team("Rolled back from v24 to v23")
  start_investigation()

Total: ~2 minutes to recover
Impact: ~10 tasks affected (0.2% of daily volume)
```

### What Went Wrong?

Debug the failure:

```
Model v24 had lower success rate. Why?

Hypothesis 1: Overfitting
  Check: Does it fail on new tasks?
  Result: Fails on both new and old tasks
  Conclusion: Not overfitting

Hypothesis 2: Training data drift
  Check: How different is training data from current?
  Result: Very different (2 weeks of data drift)
  Conclusion: Model doesn't generalize to new conditions

Hypothesis 3: Label noise
  Check: Review 100 failed tasks manually
  Result: 20% of failures were actually correct actions
  Conclusion: Training labels are noisy

Action: Improve labeling process, retrain with cleaner data
```

---

## Deployment Checklist

Before deploying any model to production:

```
☐ Model Performance
  ☐ Lab validation: Test set accuracy ≥ baseline - 2%
  ☐ No catastrophic failures: No task with >50% drop
  ☐ All task categories: No task completely broken
  ☐ Generalization: Works on unseen object variations

☐ Safety
  ☐ Safety bounds checked: Never violates joint limits
  ☐ Collision detection: Detects interference
  ☐ Emergency stop: Tested and working
  ☐ Human safety: Can't hurt person

☐ Infrastructure
  ☐ Backward compatibility: Can fall back to v23
  ☐ Storage: 500MB per robot × 100 robots fits
  ☐ Latency: < 200ms with new model
  ☐ Network: Can deploy to 100 robots in <1 hour

☐ Monitoring
  ☐ Metrics dashboard: Success rate, latency, errors
  ☐ Alerts: Configured for anomalies
  ☐ Logs: Can trace every decision
  ☐ Rollback: Can revert in <5 minutes

☐ Team Readiness
  ☐ Deployment engineer: On-call
  ☐ ML engineer: Available for questions
  ☐ Documentation: Runbooks written
  ☐ Escalation: Know who to call if problems

☐ Deployment Plan
  ☐ Deployment window: Scheduled
  ☐ Success criteria: Defined
  ☐ Failure criteria: Defined
  ☐ Communication: Notified all stakeholders
```

---

## Key Takeaways

✅ **Never deploy to all robots at once** - Always gradual (canary) or staged (blue-green)

✅ **Automate monitoring** - Don't rely on humans watching dashboards

✅ **Make rollback instant** - Not "undo and redeploy", but "switch flag"

✅ **Validate in production** - Lab metrics are good, real metrics are better

✅ **A/B test rigorously** - Use statistics, not intuition

✅ **Have a deployment checklist** - Same steps every time, no surprises

✅ **Communication is critical** - Tell everyone what you're doing

---

## Next: Operations & Maintenance

[→ Continue to Section 3: Operations & Maintenance](operations-maintenance.md)

