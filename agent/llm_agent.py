import json
import textwrap
from openai import OpenAI
from typing import Dict, Any

SYSTEM_PROMPT_NORMAL = textwrap.dedent(
    """
    You are an expert workforce scheduling agent.
    Your job: assign employees to shifts optimally.
    
    RULES (must follow):
    - Employee skills must include the shift's required_skill
    - Employee must be available on the shift's day (availability[day] == 1)
    - Employee cannot exceed max_hours_per_week
    - No two shifts for the same employee on the same (day, period)
    
    Respond with ONLY a valid JSON action object — no explanation, no markdown.
    Valid action formats:
      {"action_type": "assign", "employee_id": "emp001", "shift_id": "shf042"}
      {"action_type": "noop"}
    """
).strip()

SYSTEM_PROMPT_RECOVERY = textwrap.dedent(
    """
    You are an expert workforce scheduling agent.
    The previous approach failed. Try a completely different strategy to assign employees.
    
    RULES (must follow):
    - Employee skills must include the shift's required_skill
    - Employee must be available on the shift's day (availability[day] == 1)
    - Employee cannot exceed max_hours_per_week
    - No two shifts for the same employee on the same (day, period)
    
    Respond with ONLY a valid JSON action object — no explanation, no markdown.
    Valid action formats:
      {"action_type": "assign", "employee_id": "emp001", "shift_id": "shf042"}
      {"action_type": "noop"}
    """
).strip()

class LLMAgent:
    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name
        self.temperature = 0.0
        self.max_tokens = 150
        self.actions = []
        self.rewards = []
        self.errors = []
        
    def reset(self):
        self.actions = []
        self.rewards = []
        self.errors = []

    def _sanitize_action(self, text: str) -> dict:
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    def generate_action(self, obs_dict: Dict[str, Any], last_reward: float = None, last_error: str = None) -> dict:
        if last_reward is not None:
            self.rewards.append(last_reward)
        if last_error:
            self.errors.append(last_error)
            
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
        
        # Check recovery state
        prompt = SYSTEM_PROMPT_NORMAL
        if (len(self.rewards) >= 2 and self.rewards[-1] == 0.00 and self.rewards[-2] == 0.00) or (self.errors and self.errors[-1] is not None and self.errors[-1] != "null"):
            prompt = SYSTEM_PROMPT_RECOVERY
            
        user_content = json.dumps({
            "TASK": "Assign next employee to shift optimally.",
            "CURRENT STATE": slim,
            "PREVIOUS ACTIONS": self.actions[-3:],
            "LAST REWARD": last_reward,
            "LAST ERROR": last_error
        })

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ]
        
        attempt_text = ""
        action_dict = {"action_type": "noop"}
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            attempt_text = completion.choices[0].message.content or ""
            action_dict = self._sanitize_action(attempt_text)
            
            # Anti-repetition logic
            if action_dict in self.actions[-3:]:
                # Retry once
                messages.append({"role": "assistant", "content": attempt_text})
                messages.append({"role": "user", "content": "Do NOT repeat previous actions. Try a different strategy."})
                
                completion_retry = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False,
                )
                attempt_text = completion_retry.choices[0].message.content or ""
                action_dict = self._sanitize_action(attempt_text)
                
        except Exception:
            # Empty LLM response or invalid action -> fallback safe action
            action_dict = {"action_type": "noop"}
        
        self.actions.append(action_dict)
        # Keep memory size bounded to last 5
        if len(self.actions) > 5:
            self.actions.pop(0)
        if len(self.rewards) > 5:
            self.rewards.pop(0)
        if len(self.errors) > 5:
            self.errors.pop(0)
            
        return action_dict
