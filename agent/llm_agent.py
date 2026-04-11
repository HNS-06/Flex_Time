import json
import textwrap
from openai import OpenAI
from typing import Dict, Any

SYSTEM_PROMPT_BASE = textwrap.dedent(
    """
    You are an expert workforce scheduling agent.
    Your job: assign employees to shifts optimally.
    
    RULES (must follow):
    - Employee skills must include the shift's required_skill
    - Employee must be available on the shift's day (availability[day] == 1)
    - Employee cannot exceed max_hours_per_week
    - No two shifts for the same employee on the same (day, period)
    
    {EFFICIENCY_BIAS}
    
    Respond with ONLY a valid JSON action object — no explanation, no markdown.
    Valid action formats:
      {"action_type": "assign", "employee_id": "emp001", "shift_id": "shf042"}
      {"action_type": "noop"}
    """
).strip()

RECOVERY_BIAS = "The previous approach failed. Try a completely different strategy to assign employees."
EFFICIENCY_BIAS = "Prefer actions that move directly toward task completion in fewer steps. Avoid redundant exploration."

class LLMAgent:
    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name
        self.temperature = 0.0
        self.max_tokens = 150
        
        self.actions = []
        self.rewards = []
        self.errors = []
        
        # Elite state tracking
        self.best_reward = -float('inf')
        self.last_successful_action = None
        self.task_type = None

    def reset(self):
        self.actions = []
        self.rewards = []
        self.errors = []
        self.best_reward = -float('inf')
        self.last_successful_action = None
        self.task_type = None

    def _infer_task_type(self, obs_dict: Dict[str, Any]) -> str:
        # Task type awareness smart edge
        keys_str = str(obs_dict.keys()).lower()
        if "ui" in keys_str or "element" in keys_str or "screen" in keys_str or "viewport" in keys_str:
            return "navigation"
        return "reasoning"

    def _validate_action_text(self, text: str) -> dict:
        if not text or len(text.strip()) == 0:
            raise ValueError("Empty output from generation.")
        if len(text) > 300: # Fast proxy heuristic for detecting raw text / explanations
            raise ValueError("Output too long, likely contains explanation text.")
            
        clean_text = text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(clean_text)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format.")
            
        if not isinstance(parsed, dict) or "action_type" not in parsed:
            raise ValueError("Missing 'action_type' required key.")
            
        return parsed

    def generate_action(self, obs_dict: Dict[str, Any], last_reward: float = None, last_error: str = None) -> dict:
        # 1. Edge/Task Check
        if len(self.actions) == 0:
            self.task_type = self._infer_task_type(obs_dict)
            
        # 2. Reward-Aware Tracking / Micro Memory Biasing
        if last_reward is not None:
            self.rewards.append(last_reward)
            if self.best_reward != -float('inf') and last_reward > self.best_reward:
                self.last_successful_action = self.actions[-1] if self.actions else {"action_type": "noop"}
            if last_reward > self.best_reward:
                self.best_reward = last_reward
                
        if last_error:
            self.errors.append(last_error)
            
        # Output Compression: Minimal token structuring
        slim = {
            "unassigned": obs_dict.get("unassigned_shifts", [])[:8],
            "emps": [
                {k: e[k] for k in ("id", "skills", "availability", "assigned_hours", "max_hours_per_week")}
                for e in obs_dict.get("employees", [])
            ],
            "shifts": [
                {k: s[k] for k in ("id", "day", "period", "required_skill", "duration_hours")}
                for s in obs_dict.get("shifts", [])
                if s["id"] in obs_dict.get("unassigned_shifts", [])[:8]
            ],
        }
        
        # State Context Construct
        prompt_injections = []
        
        if self.task_type == "navigation":
            prompt_injections.append("Treat this as visual navigation: Use precise, minimal actions targeting interactive elements.")
        else:
            prompt_injections.append("Treat this as logical reasoning: Use structured rational progression.")
            
        # Strategy Bias based on Performance drop
        if last_reward is not None and last_reward < self.best_reward:
            prompt_injections.append("Avoid strategies that previously reduced reward.")
            
        # Failure Check for Context Switch
        if (len(self.rewards) >= 2 and self.rewards[-1] <= 0.00 and self.rewards[-2] <= 0.00) or (self.errors and self.errors[-1] is not None and self.errors[-1] != "null"):
            prompt_injections.append(RECOVERY_BIAS)
            
        base_prompt = SYSTEM_PROMPT_BASE.replace("{EFFICIENCY_BIAS}", EFFICIENCY_BIAS)
        compiled_prompt = base_prompt + "\n\nCRITICAL CONTEXT:\n" + "\n".join(prompt_injections)
        
        user_payload = {
            "CURRENT STATE": slim,
            "PREVIOUS ACTIONS": self.actions[-3:], # Short context to compress
        }
        if self.last_successful_action:
            user_payload["BEST PAST DIRECTIVE"] = self.last_successful_action

        messages = [
            {"role": "system", "content": compiled_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ]
        
        action_dict = {"action_type": "noop"} # Safe default mapping
        
        # Primary Action Generation Block
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            attempt_text = completion.choices[0].message.content or ""
            action_dict = self._validate_action_text(attempt_text)  # Strict validation layer
            
        except Exception as initial_err:
            # Confidence-Based Retry System -> Single fast retry loop with strict penalty prompts
            try:
                retry_messages = messages + [
                    {"role": "assistant", "content": str(attempt_text) if 'attempt_text' in locals() else "Error"},
                    {"role": "user", "content": "Return ONLY a valid executable action JSON constraint. No explanation."}
                ]
                completion_retry = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=retry_messages,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    stream=False,
                )
                action_dict = self._validate_action_text(completion_retry.choices[0].message.content or "")
            except Exception:
                # Safe Fallback Action - Double failure defaults seamlessly safely without loop crash
                action_dict = {"action_type": "noop"}
                
        # Micro-Memory tracking caps
        self.actions.append(action_dict)
        if len(self.actions) > 5: self.actions.pop(0)
        if len(self.rewards) > 5: self.rewards.pop(0)
        if len(self.errors) > 5: self.errors.pop(0)
            
        return action_dict
