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

from server.engine import FlexTimeEnv, TASK_CONFIGS
from server.models import Action
from agent.llm_agent import LLMAgent

# Mandatory environment variables with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "FlexTime"

MAX_STEPS = 120
TEMPERATURE = 0.0
MAX_TOKENS = 120


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


def run_task(agent: LLMAgent, env: FlexTimeEnv, task_id: str):
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
        
        agent.reset()
        
        last_reward = None
        last_error = None
        
        for step in range(1, cur_max_steps + 1):
            if done:
                break
                
            # Smart Early Termination
            if len(rewards) >= 3 and rewards[-1] == rewards[-2] == rewards[-3]:
                # Terminate if same reward repeats 3 times (no progress)
                done = True
                break
                
            # Predict
            action_dict = agent.generate_action(obs_dict, last_reward, last_error)
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
            
            last_reward = reward
            last_error = error
            
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
    if not HF_TOKEN:
        print("[DEBUG] HF_TOKEN is missing. This will crash. Please set HF_TOKEN.", flush=True)
        # We allow client initialization crash if token is missing as it enforces the constraint.
        
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy-key")
    env = FlexTimeEnv()
    agent = LLMAgent(client, MODEL_NAME)
    
    tasks = ["task_easy", "task_medium", "task_hard"]
    for t_id in tasks:
        run_task(agent, env, t_id)


if __name__ == "__main__":
    main()
