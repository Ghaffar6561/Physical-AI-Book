# Section 3: Operations & Maintenance

Running 100+ robots 24/7 without breaking production.

---

## The Operations Problem

### Challenge: Visibility

With 100 robots running simultaneously, how do you know:
- Which robots are working well?
- Which are struggling?
- Which will fail in the next hour?
- What's the bottleneck right now?

**Solution: Observability**

---

## Observability: The Three Pillars

### 1. Metrics (What is happening?)

```
Fleet-level metrics (check every minute):
  ├─ Overall success rate: 91.2% (↓ trending down!)
  ├─ Healthy robots: 98/100 (98% uptime)
  ├─ Avg latency: 145ms (p95: 250ms)
  ├─ Tasks completed: 12,450/day
  └─ Model version: v23 (running since 3 days)

Robot-level metrics (per robot):
  ├─ Success rate: 90-94% (individual variation)
  ├─ Task completion time: 30-120s
  ├─ CPU/GPU utilization: 45-95%
  ├─ Network latency: 50-300ms
  └─ Health: Temperature, battery, gripper pressure

Task-level metrics:
  ├─ Success rate by task (Grasp: 94%, Push: 89%, Insert: 75%)
  ├─ Task duration by task
  └─ Failure rate by task
```

### 2. Logs (Why did something happen?)

```
Every action is logged:

[2024-01-15 14:23:45] Robot-47 TASK_START task_id=5234
[2024-01-15 14:23:46] Robot-47 PERCEPTION obs_id=cam_frame_12345
[2024-01-15 14:23:47] Robot-47 DECISION action=grasp_left_arm
[2024-01-15 14:23:48] Robot-47 EXECUTION motor_cmd=0.45,0.23,0.18
[2024-01-15 14:23:49] Robot-47 FEEDBACK result=SUCCESS

[2024-01-15 14:23:50] Robot-47 TASK_START task_id=5235
[2024-01-15 14:23:51] Robot-47 PERCEPTION obs_id=cam_frame_12346
[2024-01-15 14:23:52] Robot-47 DECISION action=place_right_arm
[2024-01-15 14:23:54] Robot-47 EXECUTION motor_cmd=0.32,0.56,0.41
[2024-01-15 14:23:56] Robot-47 FEEDBACK result=FAILURE reason=gripper_slip
[2024-01-15 14:23:57] Robot-47 RECOVERY attempt_retry=true

Queryable by:
  - Time range: all events in last hour
  - Robot: all events for robot-47
  - Task: all events for task_id=5235
  - Event type: all failures
  - Custom: events where latency > 200ms
```

### 3. Traces (What was the full journey?)

```
Task 5235 journey:
  ├─ Task submitted by warehouse system (14:23:50)
  ├─ Robot-47 selected (optimal based on location)
  ├─ Model v23 runs inference (15ms)
  ├─ Decision: place_right_arm at (0.32, 0.56, 0.41)
  ├─ Safety check passes (0.1ms)
  ├─ Motor command sent (5ms)
  ├─ Motor confirms receipt (10ms)
  ├─ Gripper executes (40ms)
  ├─ Gripper fails (object too small, slipped)
  ├─ Sensor detects failure (20ms)
  ├─ Retry logic engages (automatic)
  ├─ Re-attempt with stronger grip (retry 1)
  ├─ Success! (retry 1)
  ├─ Task marked complete (14:23:56)
  └─ Duration: 6 seconds (normal: 3-4 seconds)
```

---

## Monitoring Dashboard

### Real-Time View

```
┌─────────────────────────────────────────────────┐
│  FLEET DASHBOARD - RIGHT NOW                    │
├─────────────────────────────────────────────────┤
│                                                 │
│ Status: HEALTHY ✓                              │
│ ├─ Success Rate: 91.2% (target: 90%) ✓ GREEN  │
│ ├─ Uptime: 98% (target: 99%) ⚠ YELLOW         │
│ ├─ Latency p95: 245ms (target: <250ms) ✓      │
│ └─ Model: v23 (running 3d) ✓                  │
│                                                 │
│ LIVE PERFORMANCE (last hour):                  │
│ ├─ Tasks completed: 12,450                     │
│ ├─ Tasks failed: 1,092 (8.8%)                  │
│ ├─ Avg task duration: 3.2 seconds              │
│ └─ Current throughput: 207 tasks/min           │
│                                                 │
│ ROBOT STATUS:                                  │
│ ├─ Healthy: 98/100 (98%)                      │
│ ├─ Busy: 75 robots                             │
│ ├─ Idle: 20 robots                             │
│ ├─ Maintenance: 5 robots                       │
│ └─ Offline: 0 robots ✓                         │
│                                                 │
│ ALERTS (requires attention):                   │
│ ├─ ⚠ Robot-47 success dropped to 75%           │
│ │  (usually 92%, investigate gripper)          │
│ ├─ ⚠ Insert task (15 failures in last hour)    │
│ │  (higher than baseline, check model)         │
│ └─ ✓ All other systems normal                  │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Alert Rules

```
Metric              Threshold       Action
────────────────────────────────────────────────
Fleet success       < 88%          Critical alert
Robot success       < 80%          Alert, investigate
Latency p95         > 300ms        Alert, check network
Error rate          > 5%           Alert, investigate
Robot temp          > 70°C         Warning, monitor
Gripper pressure    > 80%          Warning, maintenance
Disk space          < 10% free     Warning, cleanup
Network drops       > 10/hour       Alert, network team

Action on alert:
  - Notify on-call engineer
  - Create incident ticket
  - Start troubleshooting playbook
  - Monitor metric until resolved
```

---

## Incident Response

### Detection → Investigation → Resolution

```
INCIDENT: Success rate drops from 92% to 85%

DETECTION (automated, 2 min):
  ✓ Metric anomaly detected
  ✓ Alert sent to on-call engineer
  ✓ Slack: "Fleet success rate anomaly"

INVESTIGATION (engineer, 5 min):
  ?  Which tasks are affected?
  → All tasks (failure is model-wide, not task-specific)
  ?  Which robots are affected?
  → Robots 1-50 (switched to v24 yesterday), 51-100 (still on v23)
  ?  When did it start?
  → Started 2 hours ago
  ?  What changed?
  → v24 deployed to 50% of fleet 2 hours ago

  CONCLUSION: v24 has a bug

DECISION (5 min):
  Action: Rollback v24 to v23 on all robots
  Risk: High (customers seeing failures)
  Benefit: Restore service quality

EXECUTION (2 min):
  $ fleet.rollback_to(version='v23')
  ✓ Robots 1-50 rolled back
  ✓ Monitoring success rate...
  ✓ Success rate recovering: 85% → 88% → 91%

VERIFICATION (5 min):
  ✓ Success rate back to 92%
  ✓ No new errors
  ✓ All systems normal

POST-INCIDENT (30 min):
  - Review v24 training logs
  - Identify bug (data quality issue)
  - Fix: Retrain with cleaned data
  - Create test: Prevent future issue
  - Document: Add to runbook

CLOSURE:
  Incident resolved in 30 minutes
  Root cause: training data corruption
  Prevention: Add data validation step
```

---

## Predictive Maintenance

### The Goal

Detect problems BEFORE they become failures.

### Approach

```
Collect health signals over time:

Robot temperature:
  Day 1: Normal (25°C)
  Day 5: Warming (32°C)
  Day 10: Hot (45°C)
  Day 12: Too hot (52°C) ← Will fail tomorrow

Action: Schedule maintenance TODAY, before failure

Motor current:
  Baseline: 2.5A average
  Week 1: 2.4A (healthy)
  Week 2: 2.6A (slight increase)
  Week 3: 3.2A (worn bearing)
  Week 4: 4.5A (about to jam) ← Schedule replacement

Action: Proactive replacement costs $500
         Emergency replacement costs $5000 + downtime

Gripper grip force:
  New: Max force 95 N
  After 1000 trials: 92 N
  After 5000 trials: 85 N
  After 10000 trials: 75 N ← Maintenance needed
  After 15000 trials: <50 N ← Can't grip (failure)

Action: Every 10K trials, recalibrate
```

### Predictive Models

```
Train on historical data:
  Input:  [temperature, motor_current, gripper_pressure, uptime]
  Output: days_until_failure

Model learns:
  ✓ Temp > 50°C → ~1 day until failure
  ✓ Motor current rising 0.1A/week → ~2 weeks until failure
  ✓ Gripper pressure unstable → ~3 days until failure

Implementation:
  Every morning, run model on all robots:
    Robot-47: 0 days (FAILING NOW!)
    Robot-23: 2 days (schedule maintenance)
    Robot-99: 14 days (monitor)
    Others: >30 days (OK)

Maintenance scheduling:
  - Critical (0-1 days): Fix immediately
  - Urgent (2-3 days): Schedule this week
  - Soon (4-7 days): Schedule next week
  - Later (>7 days): No action needed
```

---

## Troubleshooting Workflows

### Decision Tree for Common Issues

```
PROBLEM: Robot stuck (not moving)

1. Is robot powered?
   NO → Power on, test
   YES → Continue

2. Is network connected?
   NO → Check WiFi, restart modem
   YES → Continue

3. Is motor responsive?
   NO → Check motor connections, restart robot
   YES → Continue

4. Does robot respond to commands?
   NO → Reboot ROS
   YES → Continue

5. Is gripper stuck?
   NO → Continue
   YES → Manually open gripper

6. Is workspace clear?
   NO → Remove obstacles
   YES → Continue

7. Did it fail on this task before?
   YES → Requeue to different robot
   NO → Investigate model (wrong action?)

RESOLUTION:
  • 90% of cases: Resolved in steps 1-6
  • 9% of cases: Requeue to different robot
  • 1% of cases: Manual intervention needed
```

### Troubleshooting Checklist

```
For LATENCY issues:
  ☐ Check network latency: ping server
  ☐ Check GPU load: nvidia-smi
  ☐ Check model size: is it too large?
  ☐ Check batch size: can we reduce?
  ☐ Check queue depth: too many pending?

For SUCCESS RATE issues:
  ☐ Check if model changed recently
  ☐ Check if environment changed (lighting, objects)
  ☐ Check if hardware worn (gripper, motor)
  ☐ Check if data changed (new object types)
  ☐ Check failure logs (categorize failures)

For CONNECTIVITY issues:
  ☐ Check WiFi signal strength
  ☐ Check network bandwidth available
  ☐ Check if server is reachable
  ☐ Check firewall rules
  ☐ Check if fallback to 4G is working

For SAFETY issues:
  ☐ Check emergency stop button
  ☐ Check safety bounds are correct
  ☐ Check collision detection is working
  ☐ Check gripper pressure is safe
  ☐ Check that humans can always intervene
```

---

## Runbooks (Step-by-Step Procedures)

### Runbook 1: Responding to High Failure Rate

```
Symptom: Success rate drops below 88%

Step 1: Verify the alert (1 min)
  $ fleet.get_success_rate(timespan='1hour')
  > 87.2% ✓ Confirmed

Step 2: Identify which tasks are failing (2 min)
  $ fleet.get_success_rate_by_task(timespan='1hour')
  Grasp:  94% (normal)
  Push:   89% (normal)
  Insert: 62% ← PROBLEM
  Others: 85-92% (mostly normal)

  Conclusion: Insert task is failing

Step 3: Check if robot or task issue (3 min)
  $ fleet.get_success_rate_by_robot(task='insert')
  Robots 1-20: 85% (normal)
  Robots 21-40: 45% ← PROBLEM GROUP
  Robots 41-60: 88% (normal)
  Robots 61-100: 90% (normal)

  Conclusion: Robots 21-40 are failing

Step 4: Investigate root cause (5 min)
  ?  Did something change on these robots?
  $ robot.get_recent_events(robot_id=21, hours=2)
  > Software update at 10:00am

  ?  Was this in the update?
  $ compare_versions('v23', 'v24')
  > Yes, Insert action changed

  Conclusion: Software update broke Insert task

Step 5: Decide on action (2 min)
  Option A: Rollback software (safest)
  Option B: Update policy for Insert task
  Option C: Restrict Insert task to robots 1-20

  Decision: Rollback software (lowest risk)

Step 6: Execute rollback (2 min)
  $ fleet.rollback_software('robots 21-40', to_version='previous')
  ✓ Rollback complete
  ✓ Monitoring success rate...
  ✓ Insert task success: 62% → 88% (recovered!)

Step 7: Monitor recovery (5 min)
  $ fleet.monitor_metrics(duration='5min')
  ✓ Success rate: 87.2% → 91.2%
  ✓ All tasks normal
  ✓ No new errors

Step 8: Document incident (10 min)
  - Create ticket: "Software update v24 broke Insert task"
  - Root cause: Policy changes not validated
  - Fix: Retrain with better validation
  - Prevention: Validate all task-specific changes before release

Total time: 30 minutes
```

---

## Maintenance Schedule

### Daily
- Review dashboard health
- Check for any alerts
- Monitor success rate trend
- Verify backups completed

### Weekly
- Review failure logs
- Analyze trends in performance
- Check for aging hardware
- Run safety tests
- Predictive maintenance check

### Monthly
- Hardware calibration
- Clean cameras and sensors
- Replace worn parts
- Full system test
- Training & knowledge review

### Quarterly
- Major software updates
- Hardware upgrades
- Performance optimization
- Safety certification
- Team training

---

## Key Takeaways

✅ **Observability is fundamental** - Can't optimize what you can't see

✅ **Alerts should be actionable** - Not "something is wrong", but "do this"

✅ **Automate everything** - No manual dashboards at 2am

✅ **Have playbooks** - Same steps every time, consistent results

✅ **Predict, don't react** - Catch issues before they impact customers

✅ **Document everything** - Future you will thank current you

---

## Next: Real-World Case Studies

[→ Continue to Section 4: Case Studies](case-studies.md)

