# Cost Analysis: Economics of Robotic Automation

When does automation become profitable?

---

## Total Cost of Ownership (TCO)

### Components of Robot Cost

```
Initial Investment:
  Robot hardware:          $50,000
  Gripper:                 $5,000
  Camera + sensors:        $3,000
  Computing (onboard GPU): $2,000
  Integration:             $10,000
  ─────────────────────────────────
  TOTAL CAPITAL:           $70,000

Recurring Costs (Annual):
  Maintenance:             $5,000
  Replacement parts:       $3,000
  Software/updates:        $2,000
  Power/network:           $1,000
  ─────────────────────────────────
  TOTAL OPEX:              $11,000/year

Personnel Costs:
  Data collection (80 hrs @ $50/hr): $4,000
  Model training (40 hrs @ $100/hr):  $4,000
  Deployment/monitoring (20 hrs):     $2,000
  ─────────────────────────────────
  TOTAL LABOR:             $10,000

TOTAL FIRST YEAR:          $91,000
TOTAL YEAR 2+:             $21,000/year
```

---

## Single Robot Economics

### The Math

```
Scenario: Manufacturing facility with 100 grasping tasks

Manual Labor:
  Current: 2 workers × $20/hour × 2000 hours/year = $80K/year
  Time: 8 hours/day, 5 days/week

Robotic Automation:
  Upfront: $91K (robot + software)
  Year 1 cost: $91K + $10K labor = $101K
  Year 2+ cost: $21K/year

Payback period:
  Savings per year: $80K (labor cost eliminated)
  Break-even: (101K - 80K) / 80K = 0.26 years = 3 months!

Return on Investment (5 years):
  Total revenue: 5 × $80K = $400K
  Total cost: $101K + 4×$21K = $185K
  Profit: $215K
  ROI: 115%
```

### Sensitivity Analysis

```
What if things go wrong?

Scenario 1: Robot success rate 70% (not 90%)
  Need second worker for 30% of tasks
  Labor: $80K × 0.3 = $24K
  Total Year 1: $101K + $24K = $125K
  Payback: Still break-even in ~2 years (marginal)

Scenario 2: Training takes 6 months (not 1 month)
  Deferred revenue: 6 months × $80K/year / 12 = $40K
  Total Year 1: $101K + $40K = $141K
  Payback: 2.5 years

Scenario 3: Robot breaks, $10K repair
  Unexpected cost but amortized over 5 years
  Impact: Minimal

Lesson: Robot usually profitable even if issues arise
```

---

## Fleet Economics: Scaling

### Single Robot vs Fleet

```
1 Robot:
  Capex: $70K
  Opex/year: $11K
  Labor/year: $10K
  Tasks: 100 (one facility)
  Production: 50 items/day

Fleet of 10 Robots:
  Capex: $700K
  Opex/year: $110K (shared infrastructure)
  Labor/year: $50K (one engineer manages all 10)
  Tasks: 1000+ (multiple facilities)
  Production: 500 items/day

Cost per robot (fleet):
  Capex: $70K (same)
  Opex: $11K (same) + $5K shared = $16K (slightly higher)

Per-item cost:
  Single robot: $91K/year ÷ (50 items/day × 250 days) = $73/item
  Fleet: $810K/year ÷ (500 items/day × 250 days) = $6.48/item

Improvement: 11× lower cost per item!
```

### Economies of Scale

```
As fleet grows from 1 to 100 robots:

Cost per robot per year:
  1 robot:    $21K (capex amortized + opex)
  10 robots:  $16K (shared ops, bulk discounts)
  50 robots:  $14K (shared infrastructure)
  100 robots: $13K (global optimization)

Factors:
  - Bulk discounts on hardware (5-10% savings at 100+)
  - Shared infrastructure (networking, tools, etc.)
  - Shared engineers (1 engineer can manage 20 robots)
  - Learning efficiency (faster to deploy new tasks)
```

---

## Data Collection Cost

### The Biggest Expense

```
For training on 150 tasks:

Method 1: Manual Teleoperation (Most Expensive)
  ├─ Expert pay: $100/hour
  ├─ Time per demo: 2 minutes
  ├─ Demos needed: 500 per task × 150 tasks = 75,000 demos
  ├─ Total time: 75,000 × 2 min = 150,000 minutes = 2,500 hours
  ├─ Cost: 2,500 × $100 = $250,000
  └─ Timeline: 6 months (one person)

Method 2: Self-Play + Domain Randomization (Cheaper)
  ├─ Robot cost: $70K
  ├─ GPU cost: $500/month
  ├─ Time: 3 months
  ├─ Total data: 500K episodes (from sim)
  ├─ Labor: 10 hours setup/monitoring = $1,000
  ├─ Total cost: $70K + $1.5K + $1K = $72.5K
  └─ Timeline: 3 months

Method 3: Crowdsourced (Cheapest)
  ├─ Worker pay: $10/hour
  ├─ Time per demo: 5 minutes
  ├─ Demos needed: 75,000
  ├─ Total time: 75,000 × 5 min / 60 = 6,250 hours
  ├─ Cost: 6,250 × $10 = $62,500
  ├─ Quality control: $10K (review bad data)
  ├─ Total: $72.5K
  └─ Timeline: 2-3 months (parallel workers)

Trade-offs:
  Teleop: High cost, high quality, slow
  Self-play: Medium cost, sim biases, fast
  Crowdsource: Low cost, quality issues, fast
```

---

## When Automation Breaks Even

### The Break-Even Graph

```
Profit/Loss ($K)
     200 │
         │
     100 │                    ╱─── (ROI positive)
         │                  ╱
       0 ├──────────────┬──────────
         │              ↓ Break-even
    -100 │           (18 months)
         │         ╱
    -200 │      ╱
         └───────────────────────→
         0   1   2   3   4   5 (Years)

Year 1: -$101K (invest in robot)
Year 2: -$80K ($21K costs - $101K invested, not yet ROI+)
Year 3: +$59K (start profiting)
Year 5: +$319K (cumulative profit)
```

### Cost vs Savings

```
Tasks per day:         50      100     200
Labor savings/year:   $40K    $80K   $160K
Break-even time:     2.5yr   1.3yr   0.6yr
Year-5 profit:       $95K    $215K   $435K

Lesson: Utilization matters!
  High-utilization tasks (200/day): Break even in 6 months
  Low-utilization (50/day): Break even in 2.5 years
```

---

## Comparison: Manual vs Automation

### Long-Term Cost

```
5-Year Total Cost of Ownership:

Manual (2 workers):
  Labor: 2 × $40K/year × 5 years = $400K
  Overhead: $100K
  TOTAL: $500K
  Success: 99% (humans are reliable)

Single Robot:
  Capex: $70K
  Opex: $11K × 5 = $55K
  Labor: $10K × 5 = $50K
  TOTAL: $175K
  Success: 85% (average robot)

Fleet of 10 Robots:
  Capex: $700K
  Opex: $15K × 10 × 5 = $750K
  Labor: $50K × 5 = $250K
  TOTAL: $1.7M
  Success: 85% per robot × 10 = 850 items vs 500 (70% higher throughput)

Cost per item (5 years):
  Manual: $500K ÷ 500 items/day ÷ 250 days ÷ 5 years = $80/item
  Single robot: $175K ÷ 50 items/day ÷ 250 days ÷ 5 = $280/item
              (Wait, this looks worse! Why use robot?)

  Answer: Humans are 80% cheaper per item, but...
    - Manual is slow (50 items/day)
    - Robot is consistent (85% success, not 99% but good enough)
    - If you need more speed, robot wins:

  Fleet of 10: $1.7M ÷ 500 items/day ÷ 250 days ÷ 5 = $27/item
                (6.5× cheaper than manual at high volume!)
```

---

## Capital Requirements

### Funding Needed

```
Scenario: Building robotic fulfillment center

Phase 1: Prototype (1 robot, 1 task)
  Budget: $100K
  Timeline: 3 months
  Goal: Prove concept works

Phase 2: Pilot (5 robots, 5 tasks)
  Budget: $500K
  Timeline: 6 months
  Goal: Demonstrate ROI, optimize costs

Phase 3: Full Deployment (50 robots, 100+ tasks)
  Budget: $5M
  Timeline: 12 months
  Goal: Scale to production

Total funding needed: $5.6M
Expected ROI (year 5): 300% (assuming $17M revenue)
```

---

## Hidden Costs (Often Overlooked)

```
Cost Category              Amount    Why
───────────────────────────────────────────
Integration labor         $10-50K   Making robot fit existing systems
Simulation development     $5-10K    Building sim for data collection
Network/infrastructure     $5-20K    WiFi, power, safety systems
Regulatory compliance      $2-10K    Safety certifications
Training staff             $5-15K    Teaching humans to work with robots
Downtime/maintenance       $5-15K    Robot breaks, unexpected repairs
Data quality assurance     $5-20K    Labeling, filtering bad data
Failure recovery           $2-5K     Emergency procedures
Version control/testing    $2-5K     ML ops infrastructure
───────────────────────────────────────────
TOTAL HIDDEN COSTS:        $41-150K
```

---

## Cost Reduction Strategies

### How to Reduce Costs

```
Strategy 1: Reduce Data Collection Cost
  Current: $250K (manual teleoperation)
  Target: $50K (self-play + sim)
  Savings: $200K
  How: Use domain randomization instead of real data

Strategy 2: Reduce Hardware Cost
  Current: $70K (research robot)
  Target: $20K (cheaper industrial manipulator)
  Savings: $50K
  Trade-off: Less flexible, but okay for single task

Strategy 3: Shared Infrastructure
  Current: 1 robot = full setup cost
  Target: 10 robots = shared setup
  Savings: $30K per robot
  Example: Shared server, single control system

Strategy 4: Reduce Labor Costs
  Current: $10K/year (model training, deployment)
  Target: $3K/year (automation, pre-trained models)
  Savings: $7K/year
  How: Use open-source models, fewer custom trainings

Total possible savings: $287K first year!
```

---

## Decision Tree: When to Automate

```
START: Considering robotic automation?

  ├─ Repetitive task?
  │  └─ NO  → Don't automate (better to hire)
  │  └─ YES → Continue
  │
  ├─ Volume? (items per day)
  │  └─ < 50   → Manual is cheaper
  │  └─ 50-200 → Break-even (depends on details)
  │  └─ >200   → Automation profitable
  │
  ├─ Success rate needed?
  │  └─ >99%  → Use humans (robots hard to get >90%)
  │  └─ 80-99% → Robots are fine
  │  └─ <80%  → Neither is acceptable
  │
  ├─ Timeline flexibility?
  │  └─ Need results in 3 months?  → Hire humans quickly
  │  └─ Can wait 6-12 months?      → Automate (will pay off)
  │
  └─ DECISION:
     └─ Automate IF: High volume (>100/day), 80%+ okay, patience for 6-12 months
     └─ Manual IF: Low volume, high accuracy needed, fast turnaround needed
```

---

## Real-World Example: Amazon Warehouse Automation

```
Scale:
  Deployed: 520,000+ robots (Kiva drive units)
  Cost per robot: ~$20K
  Total investment: $10B+

Return:
  Fulfillment speed: 5× faster
  Labor: 30% less workers needed (net: 10K jobs created elsewhere)
  Revenue impact: Enables faster shipping = more sales

ROI:
  Cost: $10B + ongoing maintenance
  Benefit: Enables $50B+ incremental revenue (estimate)
  Net: Highly profitable

Lesson: Scale is key
  Small fleet (10 robots): Questionable ROI
  Large fleet (100K robots): Massive ROI
```

---

## Key Takeaways

✅ **Single Robot**: Break-even in 1-2 years at 50+ items/day
✅ **Fleet of 10**: 11× lower cost per item than single robot
✅ **Data Collection**: Usually biggest cost ($50-250K)
✅ **Hidden Costs**: Plan for 40-150K in unforeseen expenses
✅ **Utilization**: Key driver of profitability
✅ **Scale Matters**: ROI improves dramatically with fleet size

---

## Next Steps

1. **Calculate your TCO** - What are all the costs?
2. **Estimate utilization** - How many items/day will you run?
3. **Compute break-even** - When does automation pay for itself?
4. **Compare to alternatives** - Is hiring better? Outsourcing?
5. **Build business case** - Can you justify the investment?

---

## Further Reading

- **Automation Economics** (Acemoglu & Johnson, 2018)
- **Robot Operating Cost Analysis** (DHL, 2020)
- **Manufacturing ROI** (McKinsey Global Institute)
- **Fulfillment Center Economics** (Amazon blog)

---

**Next Section:** [Diagrams & Architecture →](scaling-pipeline.md)

