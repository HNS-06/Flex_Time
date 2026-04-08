---
title: FlexTime AI
emoji: 🕐
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - scheduling
  - reinforcement-learning
  - workforce
  - optimization
  - real-world
pinned: false
license: mit
short_description: Real-world AI workforce scheduling OpenEnv environment.
---

# 🕐 FlexTime — AI Workforce Scheduling Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-22c55e?style=flat-square)](https://github.com/openenv/openenv)
[![Python](https://img.shields.io/badge/Python-3.11-3b82f6?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-22c55e?style=flat-square)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square)](LICENSE)

---

**FlexTime** is a fully-featured [OpenEnv](https://github.com/openenv/openenv)-compliant environment
where AI agents learn to solve the **real-world workforce scheduling problem**: assigning employees
to shifts while satisfying hard operational constraints (skill matching, availability, maximum hours)
and optimizing soft objectives (fairness, employee preferences, demand coverage).

This is a problem that operations managers in **retail, healthcare, logistics, and hospitality** face
every week — affecting millions of workers worldwide. FlexTime models it faithfully, making it both
a genuine benchmark and a practical tool for developing AI scheduling assistants.

---

## 🗺 Environment Overview

| Property | Value |
|---|---|
| **Domain** | Workforce Scheduling / Operations Research |
| **Tasks** | 3 (Easy → Medium → Hard) |
| **Episode horizon** | 20 / 60 / 120 steps |
| **Reward** | Dense, shaped (–1.0 to +1.0) |
| **Action space** | Discrete: assign, remove, swap, noop |
| **Observation** | Structured JSON: employees, shifts, assignments, metrics |
| **Constraints** | 4 hard (H1–H4) + 4 soft (S1–S4) |
| **Baseline agent** | Greedy (rule-based) + LLM (OpenAI API) |

---

## ⚡ Quick Start

### Docker

```bash
git clone https://huggingface.co/spaces/your-org/flextime
cd flextime

docker build -t flextime .
docker run -p 7860:7860 flextime

# Environment live at http://localhost:7860
```

### Local (no Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

### Run Baseline

```bash
# Rule-based greedy baseline (no API key needed)
python -m scripts.baseline

# LLM-based baseline (requires OpenAI key)
export OPENAI_API_KEY=sk-...
python -m scripts.baseline --llm --model gpt-4o-mini

# Single task
python -m scripts.baseline --task task_hard
```

---

## 📋 Task Definitions

### Task 1 — Basic Shift Coverage (Easy)
**Target score: 1.0**

Assign employees to 5 open morning shifts for a single day. All employees are available;
skills match every shift. The agent needs to fill all slots without creating overlaps.

```
Employees: 5    Shifts: 5    Max steps: 20
Constraints: H1 (no overlap), H2 (skill match)
```

### Task 2 — Weekly Schedule with Constraints (Medium)
**Target score: 0.85**

Build a complete weekly schedule for 8 employees across 30 shifts. Must respect:
- Skill requirements (cashier, supervisor, inventory)
- Availability windows per employee per day
- 40-hour maximum working week per employee
- Soft fairness distribution (±4h between min/max hours)

```
Employees: 8    Shifts: 30    Max steps: 60
Constraints: H1–H4 (hard) + S1–S2 (soft)
```

### Task 3 — Fair Optimization Under Pressure (Hard)
**Target score: 0.75 on ALL sub-metrics simultaneously**

The hardest task: 12 employees, 50 shifts, 3 pre-seeded conflicts to resolve.
The grader applies a joint threshold — **all** of coverage, fairness, constraint satisfaction,
and demand satisfaction must exceed 0.75 simultaneously. Missing any single threshold
triggers a penalty: `score = min(sub_scores) × 0.9`.

```
Employees: 12    Shifts: 50    Max steps: 120
Constraints: H1–H4 + S1–S4 (all 8 active)
Pre-seeded conflicts: 3
Skills: 5 (cashier, supervisor, inventory, customer_service, technician)
```

---

## 🔌 API Reference

All endpoints are OpenEnv-compliant. Interactive docs at `/docs`.

### Core Endpoints

#### `POST /reset`
Initialize or reset the environment for a given task.

```json
// Request
{ "task_id": "task_medium", "seed": 42 }

// Response: Observation
{
  "week_id": "week-a3f912",
  "task_id": "task_medium",
  "employees": [...],
  "shifts": [...],
  "assignments": [],
  "unassigned_shifts": ["shf001", "shf002", ...],
  "conflicts": [],
  "metrics": {
    "total_shifts": 30,
    "assigned_shifts": 0,
    "coverage_rate": 0.0,
    "hard_violations": 0,
    "fairness_delta": 0.0,
    "fairness_score": 1.0,
    "demand_satisfaction": 0.0,
    "preference_satisfaction": 0.0
  },
  "done": false,
  "step_count": 0,
  "max_steps": 60
}
```

#### `POST /step`
Apply an action. Returns new observation, shaped reward, done flag, and info dict.

```json
// Request — Action
{
  "action_type": "assign",       // assign | remove | swap | noop
  "employee_id": "emp001",       // required for assign/remove/swap
  "shift_id": "shf042",          // required for assign/remove
  "target_employee_id": "emp005" // required for swap only
}

// Response: StepResult
{
  "observation": { ... },        // full Observation
  "reward": {
    "total": 0.15,
    "components": {
      "shift_covered": 0.225,
      "demand_signal": 0.043,
      "constraint_violated": 0.0,
      "fairness": 0.02
    },
    "info": { "result": "action=assign, Δcoverage=+0.033, hard_violations=0" }
  },
  "done": false,
  "info": { "step": "1", "coverage": "0.033" }
}
```

#### `GET /state`
Returns current observation without changing state.

#### `GET /tasks`
Returns all tasks with descriptions, schemas, and expected difficulty.

#### `GET /grader`
Returns normalized score (0.0–1.0) for the current episode.

```json
{
  "task_id": "task_medium",
  "score": 0.842,
  "breakdown": {
    "coverage_score": 0.9333,
    "fairness_score": 0.875,
    "constraint_score": 0.9,
    "demand_score": 0.891,
    "preference_score": 0.723
  },
  "passed": true,
  "summary": "Score 0.8420 (PASS) — Coverage 93.3%, Fairness 87.5%, Hard violations: 1"
}
```

#### `POST /baseline`
Runs the baseline agent against all 3 tasks and returns reproducible scores.

```json
// Query param: ?use_llm=true (requires OPENAI_API_KEY)
{
  "model": "GreedyBaseline",
  "results": [...],
  "mean_score": 0.717,
  "timestamp": "2026-03-30T12:00:00Z"
}
```

---

## 📦 Observation Space

```python
class Observation(BaseModel):
    week_id: str                    # Unique episode identifier
    task_id: str                    # Which task is active
    employees: List[Employee]       # Full roster with skills, availability, hours
    shifts: List[Shift]             # All shifts with day, period, skill, demand
    assignments: List[Dict]         # Active assignments [{employee_id, shift_id, hours}]
    unassigned_shifts: List[str]    # Shift IDs still needing coverage
    conflicts: List[ConstraintViolation]  # Active violations with type and severity
    metrics: ScheduleMetrics        # Computed KPIs
    done: bool                      # Episode termination flag
    step_count: int
    max_steps: int
```

---

## 🎮 Action Space

```python
class Action(BaseModel):
    action_type: Literal["assign", "remove", "swap", "noop"]
    employee_id: Optional[str]       # emp001 ... emp012
    shift_id: Optional[str]          # shf001 ... shf050
    target_employee_id: Optional[str] # For swap only
```

---

## 🏆 Reward Function

Dense shaped reward at every step. Range: **[–1.0, +1.0]**

| Component | Signal | Value |
|---|---|---|
| `shift_covered` | New shift filled | +0.15 × Δcoverage |
| `demand_signal` | Demand-weighted coverage | +0.05 × Δdemand |
| `constraint_violated` | New hard violation | –0.20 per violation |
| `constraint_resolved` | Violation removed | +0.10 per resolution |
| `fairness` | Fairness score improved | +0.03 × Δfairness |
| `conflict_resolved` | Pre-seeded conflict fixed | +0.10 bonus |
| `invalid_action` | Non-existent IDs, etc. | –0.05 |
| `noop` | No-operation | 0.0 |

---

## 🔒 Constraint System

### Hard Constraints (must not be violated)
| ID | Name | Description |
|---|---|---|
| H1 | No Overlapping Shifts | One shift per employee per (day, period) |
| H2 | Skill–Role Match | Employee skills ⊇ shift required_skill |
| H3 | Max Hours | Σ hours ≤ max_hours_per_week (typically 40h) |
| H4 | Availability | Employee must be available on shift day |

### Soft Constraints (penalized in objective)
| ID | Name | Description |
|---|---|---|
| S1 | Fair Distribution | max(hours) – min(hours) ≤ 4h |
| S2 | Preference Matching | Assign preferred shift period when possible |
| S3 | Min 11h Rest Gap | ≥ 11h between consecutive shifts |
| S4 | Consecutive Days | ≤ 5 consecutive working days |

---

## 📊 Baseline Scores

Reproducible scores with seed=42, GreedyBaseline agent:

| Task | Score | Steps | Pass? |
|---|---|---|---|
| task_easy | ~0.95 | ~5 | ✅ |
| task_medium | ~0.72 | ~28 | ❌ (target 0.85) |
| task_hard | ~0.48 | ~52 | ❌ (target 0.75) |
| **Mean** | **~0.72** | | |

The gap between greedy (~0.72) and target (~0.85) on medium/hard provides excellent
learning signal for RL agents.

---

## 🧪 Testing

```bash
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_environment.py::TestSpecCompliance -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

---

## 📁 Project Structure

```
flextime/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application, all endpoints
│   ├── engine.py        # Core environment: state, step, reward, grader
│   ├── models.py        # Pydantic typed models (Observation, Action, Reward)
│   └── static/
│       └── index.html   # Interactive demo UI
├── scripts/
│   ├── __init__.py
│   └── baseline.py      # Greedy + LLM baseline agents & CLI
├── tests/
│   └── test_environment.py  # Full pytest test suite
├── openenv.yaml         # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🤗 HuggingFace Spaces Deployment

1. Create a new HF Space: **Docker** template, tagged `openenv`
2. Push this repo to the Space
3. The Space will auto-build and serve at `https://huggingface.co/spaces/your-org/flextime`

The HF Space will automatically:
- Build the Docker image
- Expose port 7860
- Respond to `reset()` pings for OpenEnv validation

---

## 🧠 Why Workforce Scheduling?

Workforce scheduling is a **genuine, high-value operations problem**:

- **Scale**: Affects billions of shift workers globally (retail, healthcare, logistics)
- **Complexity**: Multi-constraint combinatorial optimization (NP-hard in general)
- **Real cost**: Poor scheduling → $millions in overtime, turnover, and burnout
- **AI gap**: Existing tools are rule-based; LLM/RL agents could outperform dramatically
- **Fairness stakes**: Biased scheduling has real worker welfare consequences

FlexTime provides the first OpenEnv environment in this domain, enabling the community
to benchmark and train agents on a problem with immediate real-world deployment value.

---

## 📄 License

MIT © FlexTime Team
