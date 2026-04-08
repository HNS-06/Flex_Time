"""
FlexTime — Baseline Inference Script
======================================
Two baseline agents:
  1. GreedyBaseline  — rule-based, no API key required (DEFAULT)
  2. LLMBaseline     — OpenAI API client, reads OPENAI_API_KEY from env

Usage (CLI):
  python -m scripts.baseline                        # greedy, all 3 tasks
  python -m scripts.baseline --llm                  # LLM agent
  python -m scripts.baseline --task task_hard       # single task
  python -m scripts.baseline --seed 0               # different seed

Called by POST /baseline endpoint in app/main.py.
Produces reproducible scores: seed=42 always gives same result.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Ensure project root is importable when run as script or module
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.engine import FlexTimeEnv, TASK_CONFIGS
from server.models import Action


# ──────────────────────────────────────────────────────────────
#  GREEDY BASELINE AGENT
#  Priority heuristic: skill match → availability → not over hours
#  → fewest assigned hours (fairness) → preferred shift match
# ──────────────────────────────────────────────────────────────

class GreedyAgent:
    """
    Rule-based greedy agent. No API key required.
    Deterministic given the same seed — guarantees reproducible scores.
    """

    name = "GreedyBaseline"

    def act(self, obs_dict: Dict) -> Dict:
        unassigned = obs_dict.get("unassigned_shifts", [])
        if not unassigned:
            return {"action_type": "noop"}

        shifts   = {s["id"]: s for s in obs_dict["shifts"]}
        employees = obs_dict["employees"]

        # Track already-assigned (day, period) slots per employee to detect overlaps
        emp_slots: Dict[str, set] = {e["id"]: set() for e in employees}
        for s in obs_dict["shifts"]:
            eid = s.get("assigned_employee_id")
            if eid:
                emp_slots.setdefault(eid, set()).add((s["day"], s["period"]))

        for shift_id in unassigned:
            shf = shifts.get(shift_id)
            if not shf:
                continue

            skill    = shf["required_skill"]
            day      = shf["day"]
            period   = shf["period"]
            duration = shf["duration_hours"]

            candidates = []
            for emp in employees:
                eid = emp["id"]
                # Hard: skill match
                if skill not in emp["skills"]:
                    continue
                # Hard: availability
                if not emp["availability"][day]:
                    continue
                # Hard: max hours
                if emp["assigned_hours"] + duration > emp["max_hours_per_week"]:
                    continue
                # Hard: no overlap on same (day, period)
                if (day, period) in emp_slots.get(eid, set()):
                    continue

                fairness_score = -emp["assigned_hours"]          # fewer hours → better
                pref_bonus     = 0.5 if emp.get("preferred_shift") == period else 0.0
                candidates.append((fairness_score + pref_bonus, eid))

            if candidates:
                candidates.sort(reverse=True)
                return {
                    "action_type": "assign",
                    "employee_id": candidates[0][1],
                    "shift_id": shift_id,
                }

        # No valid assignment found for any unassigned shift → noop
        return {"action_type": "noop"}


# ──────────────────────────────────────────────────────────────
#  LLM BASELINE AGENT  (OpenAI API client)
# ──────────────────────────────────────────────────────────────

class LLMAgent:
    """
    LLM-based agent using the OpenAI API client.
    Credentials read from OPENAI_API_KEY environment variable.
    Falls back to GreedyAgent if API key not set or call fails.
    """

    SYSTEM_PROMPT = """You are an expert workforce scheduling agent.
Your job: assign employees to shifts optimally.

RULES (must follow):
- Employee skills must include the shift's required_skill
- Employee must be available on the shift's day (availability[day] == 1)
- Employee cannot exceed max_hours_per_week
- No two shifts for the same employee on the same (day, period)

You receive the current schedule state as JSON.
Respond with ONLY a valid JSON action object — no explanation, no markdown.

Valid formats:
  {"action_type": "assign", "employee_id": "emp001", "shift_id": "shf042"}
  {"action_type": "noop"}
"""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI  # raises ImportError if not installed
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Export it before running with --llm."
            )
        self.client = OpenAI(api_key=api_key)
        self.model  = model
        self.name   = f"LLM ({model})"
        self._greedy_fallback = GreedyAgent()

    def act(self, obs_dict: Dict) -> Dict:
        # Trim observation to fit context window
        slim = {
            "unassigned_shifts": obs_dict["unassigned_shifts"][:8],
            "employees": [
                {k: e[k] for k in
                 ("id","name","skills","availability","assigned_hours","max_hours_per_week","preferred_shift")}
                for e in obs_dict["employees"]
            ],
            "shifts": [
                {k: s[k] for k in ("id","day","period","required_skill","duration_hours")}
                for s in obs_dict["shifts"]
                if s["id"] in obs_dict["unassigned_shifts"][:8]
            ],
        }
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",  "content": self.SYSTEM_PROMPT},
                    {"role": "user",    "content": json.dumps(slim)},
                ],
                max_tokens=80,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json","").replace("```","").strip()
            return json.loads(raw)
        except Exception as exc:
            print(f"[LLMAgent] API error ({exc}), falling back to greedy.")
            return self._greedy_fallback.act(obs_dict)


# ──────────────────────────────────────────────────────────────
#  EPISODE RUNNER
# ──────────────────────────────────────────────────────────────

def run_episode(env: FlexTimeEnv, agent, task_id: str, seed: int = 42) -> Dict:
    """Run one full episode. Returns grader result + episode stats."""
    obs      = env.reset(task_id=task_id, seed=seed)
    obs_dict = obs.model_dump()

    total_reward = 0.0
    steps        = 0
    noop_streak  = 0

    while not obs_dict.get("done", False):
        action_dict = agent.act(obs_dict)
        action      = Action(**action_dict)
        result      = env.step(action)
        obs_dict    = result.observation.model_dump()
        total_reward += result.reward.total
        steps        += 1

        if action.action_type == "noop":
            noop_streak += 1
            if noop_streak >= 5:
                break          # agent is stuck, stop wasting steps
        else:
            noop_streak = 0

    grade = env.grade()
    grade["episode_reward"] = round(total_reward, 4)
    grade["steps_used"]     = steps
    grade["task_id"]        = task_id
    return grade


# ──────────────────────────────────────────────────────────────
#  ASYNC RUNNER — called by POST /baseline
# ──────────────────────────────────────────────────────────────

async def run_baseline(use_llm: bool = False) -> Dict:
    """
    Run baseline agent on all 3 tasks.
    Called by POST /baseline endpoint in app/main.py.
    Returns reproducible results with seed=42.
    """
    if use_llm:
        try:
            agent = LLMAgent()
        except (ImportError, ValueError) as exc:
            print(f"[run_baseline] LLM unavailable ({exc}), using GreedyBaseline.")
            agent = GreedyAgent()
    else:
        agent = GreedyAgent()

    env     = FlexTimeEnv()
    results = []

    for task_id in TASK_CONFIGS:
        t0     = time.time()
        result = run_episode(env, agent, task_id, seed=42)
        result["elapsed_seconds"] = round(time.time() - t0, 3)
        results.append(result)
        print(f"[Baseline] {task_id}: score={result['score']:.4f}  "
              f"steps={result['steps_used']}  elapsed={result['elapsed_seconds']}s")

    mean_score = round(sum(r["score"] for r in results) / len(results), 4)

    return {
        "model":      agent.name if hasattr(agent, "name") else "GreedyBaseline",
        "seed":       42,
        "results":    results,
        "mean_score": mean_score,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
#  python -m scripts.baseline [--llm] [--task TASK] [--seed N]
# ──────────────────────────────────────────────────────────────

async def _cli_async():
    parser = argparse.ArgumentParser(
        description="FlexTime Baseline Inference — reproducible scores on all 3 tasks"
    )
    parser.add_argument("--llm",   action="store_true", help="Use OpenAI LLM agent")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (with --llm)")
    parser.add_argument("--task",  default=None, choices=list(TASK_CONFIGS),
                        help="Run a single task only")
    parser.add_argument("--seed",  type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()

    # Build agent
    if args.llm:
        try:
            agent = LLMAgent(model=args.model)
        except (ImportError, ValueError) as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
    else:
        agent = GreedyAgent()

    env          = FlexTimeEnv()
    tasks_to_run = [args.task] if args.task else list(TASK_CONFIGS)
    results      = []

    print(f"\n{'='*62}")
    print(f"  FlexTime Baseline  |  Agent: {agent.name}  |  Seed: {args.seed}")
    print(f"{'='*62}\n")

    for task_id in tasks_to_run:
        cfg = TASK_CONFIGS[task_id]
        print(f"  [{task_id}]  {cfg['name']}  ({cfg['difficulty']})")
        t0     = time.time()
        result = run_episode(env, agent, task_id, seed=args.seed)
        elapsed = round(time.time() - t0, 3)

        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"    Score:      {result['score']:.4f}  {status}  "
              f"(target ≥ {cfg['target_score']})")
        print(f"    Steps:      {result['steps_used']} / {cfg['max_steps']}")
        print(f"    Ep Reward:  {result['episode_reward']:.4f}")
        print(f"    Elapsed:    {elapsed}s")
        print(f"    Breakdown:")
        for k, v in result["breakdown"].items():
            bar = "█" * int(v * 20)
            print(f"      {k:<26} {v:.4f}  {bar}")
        print()
        result["elapsed_seconds"] = elapsed
        results.append(result)

    mean = round(sum(r["score"] for r in results) / len(results), 4)
    print(f"{'='*62}")
    print(f"  Mean Score: {mean:.4f}")
    print(f"{'='*62}\n")

    output = {
        "model":      agent.name,
        "seed":       args.seed,
        "results":    results,
        "mean_score": mean,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(output, indent=2))


def main():
    asyncio.run(_cli_async())


if __name__ == "__main__":
    main()
