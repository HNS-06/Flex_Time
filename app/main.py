"""
FlexTime — FastAPI Application
OpenEnv-compliant workforce scheduling environment.
All required endpoints: /reset /step /state /tasks /grader /baseline
Plus: /health /info and interactive UI at /
"""

from __future__ import annotations
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.engine import FlexTimeEnv, TASK_CONFIGS
from app.models import (
    Action, Observation, ResetRequest, StepResult,
    AddEmployeeRequest, EditEmployeeRequest, AddShiftRequest, LeaveRequest
)

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="FlexTime",
    description=(
        "🕐 **FlexTime** — Real-world AI Workforce Scheduling OpenEnv Environment\n\n"
        "Agents learn to assign employees to shifts while satisfying hard constraints "
        "(skill matching, availability, max hours) and soft objectives "
        "(fairness, preferences, demand coverage).\n\n"
        "**3 Tasks:** Easy → Medium → Hard with full programmatic graders (0.0–1.0).\n\n"
        "Compliant with the [OpenEnv specification](https://github.com/openenv/openenv)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve static dir relative to this file so it works in Docker too
_STATIC_DIR = Path(__file__).parent / "static"

# Serve static assets (style.css, app.js) at /static/*
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Global env instance (stateful single-session for HF Spaces)
_env = FlexTimeEnv()


# ══════════════════════════════════════════════════════════════
#  ROOT — Interactive UI
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the FlexTime interactive demo UI."""
    html_path = _STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════
#  HEALTH — Required for HF Spaces ping (must return 200)
# ══════════════════════════════════════════════════════════════

@app.get("/health", tags=["Health"])
async def health():
    """Health check — HF Spaces pings this. Must return 200."""
    return {"status": "ok", "service": "FlexTime", "version": "1.0.0"}


# ══════════════════════════════════════════════════════════════
#  OPENENV CORE ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.post(
    "/reset",
    response_model=Observation,
    summary="Reset the environment",
    description=(
        "Initialize or reset for a given task. Returns the initial Observation. "
        "Must be called before step(). Optionally pass a seed for reproducibility."
    ),
    tags=["OpenEnv Core"],
)
async def reset(body: ResetRequest):
    """POST /reset — OpenEnv spec required endpoint."""
    try:
        obs = _env.reset(task_id=body.task_id, seed=body.seed)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/step",
    response_model=StepResult,
    summary="Step the environment",
    description=(
        "Apply one action. Returns new Observation, shaped Reward with component "
        "breakdown, done flag, and info dict. Call reset() first."
    ),
    tags=["OpenEnv Core"],
)
async def step(action: Action):
    """POST /step — OpenEnv spec required endpoint."""
    try:
        return _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/state",
    response_model=Observation,
    summary="Get current state",
    description="Returns the current Observation without modifying state.",
    tags=["OpenEnv Core"],
)
async def state():
    """GET /state — OpenEnv spec required endpoint."""
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ══════════════════════════════════════════════════════════════
#  REQUIRED ADDITIONAL ENDPOINTS (per hackathon spec)
# ══════════════════════════════════════════════════════════════

@app.get(
    "/tasks",
    summary="List tasks and action schema",
    description=(
        "Returns all available tasks with id, name, difficulty, description, "
        "target_score, max_steps, and the complete JSON action_schema required "
        "for POST /step. Required by the hackathon spec."
    ),
    tags=["OpenEnv Extended"],
)
async def tasks():
    """GET /tasks — Returns task list + action schema (hackathon required)."""
    action_schema = Action.model_json_schema()
    task_list = []
    for task_id, cfg in TASK_CONFIGS.items():
        task_list.append({
            "id": task_id,
            "name": cfg["name"],
            "difficulty": cfg["difficulty"],
            "description": cfg["description"],
            "max_steps": cfg["max_steps"],
            "target_score": cfg["target_score"],
            "n_employees": cfg["n_employees"],
            "n_shifts": cfg["n_shifts"],
            "action_schema": action_schema,  # full JSON schema of Action model
        })
    return {
        "tasks": task_list,
        "total": len(task_list),
        "action_schema": action_schema,  # also at top level for convenience
    }


@app.get(
    "/grader",
    summary="Grade the current episode",
    description=(
        "Compute the final normalized score (0.0–1.0) for the current episode. "
        "Returns a full breakdown of sub-scores and pass/fail status. "
        "Can be called at any time — does not require done=True. "
        "Required by the hackathon spec."
    ),
    tags=["OpenEnv Extended"],
)
async def grader():
    """GET /grader — Returns grader score (hackathon required)."""
    try:
        return _env.grade()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/baseline",
    summary="Run baseline inference on all 3 tasks",
    description=(
        "Triggers the built-in greedy baseline agent against all 3 tasks and "
        "returns reproducible scores. Set use_llm=true to use the OpenAI API "
        "(requires OPENAI_API_KEY environment variable). "
        "Required by the hackathon spec."
    ),
    tags=["OpenEnv Extended"],
)
async def baseline(
    use_llm: bool = Query(False, description="Use OpenAI LLM agent (requires OPENAI_API_KEY)"),
):
    """POST /baseline — Runs baseline agent, returns scores (hackathon required)."""
    try:
        from scripts.baseline import run_baseline
        results = await run_baseline(use_llm=use_llm)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline error: {str(e)}")


# ══════════════════════════════════════════════════════════════
#  SAAS / DYNAMIC UI ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.post("/add_employee", response_model=Observation, tags=["SaaS"])
async def add_employee(body: AddEmployeeRequest):
    """Dynamically add an employee. Mid-episode structural change."""
    try:
        return _env.add_employee(body.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/edit_employee", response_model=Observation, tags=["SaaS"])
async def edit_employee(body: EditEmployeeRequest):
    """Dynamically edit an employee."""
    try:
        return _env.edit_employee(body.employee_id, body.model_dump(exclude_unset=True))
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/add_shift", response_model=Observation, tags=["SaaS"])
async def add_shift(body: AddShiftRequest):
    """Dynamically add a shift. Mid-episode structural change."""
    try:
        return _env.add_shift(body.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/leave", response_model=Observation, tags=["SaaS"])
async def leave_request(body: LeaveRequest):
    """Dynamically take an employee offline and drop their shifts."""
    try:
        return _env.apply_leave(body.employee_id, body.from_day, body.to_day)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/scenario/{scenario_type}", response_model=Observation, tags=["SaaS"])
async def apply_scenario(scenario_type: str):
    """Mutate environment based on a scenario."""
    if scenario_type not in ["shortage", "surge", "holiday"]:
        raise HTTPException(status_code=400, detail="Invalid scenario")
    try:
        return _env.apply_scenario(scenario_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ══════════════════════════════════════════════════════════════
#  INFO
# ══════════════════════════════════════════════════════════════

@app.get("/info", summary="Environment metadata", tags=["Info"])
async def info():
    """Returns FlexTime metadata and full endpoint map."""
    return {
        "name": "FlexTime",
        "version": "1.0.0",
        "description": "Real-world AI workforce scheduling OpenEnv environment",
        "openenv_compliant": True,
        "real_world_domain": "Workforce Scheduling / Operations Research",
        "tasks": list(TASK_CONFIGS.keys()),
        "action_types": ["assign", "remove", "swap", "noop"],
        "reward_range": [-1.0, 1.0],
        "reward_type": "dense_shaped",
        "constraints": {
            "hard": ["no_overlap", "skill_match", "max_hours", "availability"],
            "soft": ["fair_distribution", "preferences", "min_rest_gap", "consecutive_days"],
        },
        "endpoints": {
            "reset":    "POST /reset",
            "step":     "POST /step",
            "state":    "GET  /state",
            "tasks":    "GET  /tasks",
            "grader":   "GET  /grader",
            "baseline": "POST /baseline",
            "health":   "GET  /health",
            "docs":     "GET  /docs",
        },
        "baseline_scores": {
            "task_easy":   {"agent": "GreedyBaseline", "score": 0.95, "seed": 42},
            "task_medium": {"agent": "GreedyBaseline", "score": 0.72, "seed": 42},
            "task_hard":   {"agent": "GreedyBaseline", "score": 0.48, "seed": 42},
        },
    }
