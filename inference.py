"""
FlexTime Inference Baseline Script
Required by Hackathon Spec
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

# Ensure the root dir is in path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from app.engine import FlexTimeEnv, TASK_CONFIGS
from app.models import Action

# Mandatory environment variables with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "FlexTime"

MAX_STEPS = 120
TEMPERATURE = 0.0
MAX_TOKENS = 120

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert workforce scheduling agent.
    Your job: assign employees to shifts optimally.
    
    RULES (must follow):
    - Employee skills must include the shift's required_skill
    - Employee must be available on the shift's day (availability[day] == 1)
    - Employee cannot exceed max_hours_per_week
    - No two shifts for the same employee on the same (day, period)
    
    You receive the current schedule state as JSON.
    Respond with ONLY a valid JSON action object — no explanation, no markdown.
    
    Valid action formats:
      {"action_type": "assign", "employee_id": "emp001", "shift_id": "shf042"}
      {"action_type": "noop"}
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


def get_model_action(client: OpenAI, obs_dict: dict) -> dict:
    # Trim Observation to ensure it fits context and removes noisy metrics
    slim = {
        "unassigned_shifts": obs_dict["unassigned_shifts"][:8],
        "employees": [
            {k: e[k] for k in
             ("id", "name", "skills", "availability", "assigned_hours", "max_hours_per_week", "preferred_shift")}
            for e in obs_dict["employees"]
        ],
        "shifts": [
            {k: s[k] for k in ("id", "day", "period", "required_skill", "duration_hours")}
            for s in obs_dict["shifts"]
            if s["id"] in obs_dict["unassigned_shifts"][:8]
        ],
    }
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(slim)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as exc:
        # Emit an error logically, but fallback to noop to preserve the run bounds instead of crashing
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Note: If no token is provided, this handles graceful skip
        return {"action_type": "noop"}

def run_task(client: OpenAI, env: FlexTimeEnv, task_id: str):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    
    try:
        # Reset the environment for the specific task using a consistent seed
        obs = env.reset(task_id=task_id, seed=42)
        obs_dict = obs.model_dump()
        done = False
        
        cfg = TASK_CONFIGS[task_id]
        cur_max_steps = min(MAX_STEPS, cfg["max_steps"])
        target_score = cfg["target_score"]
        
        for step in range(1, cur_max_steps + 1):
            if done:
                break
            
            # Predict
            action_dict = get_model_action(client, obs_dict)
            action_str = json.dumps(action_dict).replace(' ', '')
            
            # Execute
            try:
                action = Action(**action_dict)
                result = env.step(action)
                obs_dict = result.observation.model_dump()
                reward = result.reward.total or 0.0
                done = result.done
                error = None
            except Exception as e:
                # Execution failed due to malformed action
                reward = -0.05
                done = False
                error = str(e).replace(' ', '_') # Replace spaces just in case format is very strictly space-delimited
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
        
        # Grading
        grade = env.grade()
        score = grade.get("score", 0.0)
        score = min(max(score, 0.0), 1.0) # Clamp 0-1
        success = score >= target_score
        
    except Exception as overall_e:
        print(f"[DEBUG] Overall environment failure: {overall_e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = FlexTimeEnv()
    
    tasks = ["task_easy", "task_medium", "task_hard"]
    for t_id in tasks:
        run_task(client, env, t_id)


if __name__ == "__main__":
    main()
