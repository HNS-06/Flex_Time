"""
FlexTime — Core Pydantic models.
All Observation, Action, Reward, and supporting types are fully typed
to comply with the OpenEnv specification.
"""

from __future__ import annotations
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import uuid


# ──────────────────────────────────────────────────────────────
#  DOMAIN PRIMITIVES
# ──────────────────────────────────────────────────────────────

SKILL_TYPES = Literal[
    "cashier", "supervisor", "inventory", "customer_service",
    "pharmacist", "nurse", "driver", "technician"
]

SHIFT_PERIOD = Literal["morning", "afternoon", "night"]

ACTION_TYPES = Literal["assign", "remove", "swap", "noop"]


class Employee(BaseModel):
    """An employee in the scheduling roster."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    skills: List[str] = Field(..., description="List of skill identifiers this employee holds")
    availability: List[int] = Field(
        ..., description="Binary availability per day [Mon..Sun], e.g. [1,1,0,1,1,0,0]"
    )
    max_hours_per_week: int = Field(40, ge=0, le=80)
    assigned_hours: float = Field(0.0, ge=0.0)
    preferred_shift: Optional[str] = Field(None, description="morning | afternoon | night | None")
    preference_weight: float = Field(1.0, ge=0.0, le=2.0, description="How strongly to weight this employee's preferences")

    @property
    def remaining_hours(self) -> float:
        return max(0.0, self.max_hours_per_week - self.assigned_hours)

    @property
    def is_overloaded(self) -> bool:
        return self.assigned_hours > self.max_hours_per_week


class Shift(BaseModel):
    """A single shift slot that needs to be covered."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    day: int = Field(..., ge=0, le=6, description="Day index: 0=Mon … 6=Sun")
    period: str = Field(..., description="morning | afternoon | night")
    duration_hours: float = Field(8.0, gt=0)
    required_skill: str = Field(..., description="Skill required for this shift")
    demand_level: float = Field(1.0, ge=0.0, le=3.0, description="Relative importance / demand weight")
    assigned_employee_id: Optional[str] = Field(None)

    @property
    def is_assigned(self) -> bool:
        return self.assigned_employee_id is not None

    @property
    def day_name(self) -> str:
        return ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][self.day]


class ConstraintViolation(BaseModel):
    """Represents a detected constraint violation."""
    violation_type: Literal["overlap", "skill_mismatch", "max_hours", "unavailable", "fairness"]
    employee_id: Optional[str] = None
    shift_id: Optional[str] = None
    description: str
    severity: Literal["hard", "soft"]
    penalty: float = 0.0


class ScheduleMetrics(BaseModel):
    """Computed metrics for the current schedule state."""
    total_shifts: int
    assigned_shifts: int
    coverage_rate: float = Field(..., ge=0.0, le=1.0)
    hard_violations: int
    soft_violations: int
    fairness_delta: float = Field(..., description="Max − Min assigned hours across employees")
    fairness_score: float = Field(..., ge=0.0, le=1.0)
    demand_satisfaction: float = Field(..., ge=0.0, le=1.0, description="Demand-weighted coverage")
    preference_satisfaction: float = Field(..., ge=0.0, le=1.0)
    avg_hours: float = Field(0.0, description="Average assigned hours per employee")
    unmet_demand: int = Field(0, description="Number of shifts still unassigned")
    step_count: int
    episode_reward: float


# ──────────────────────────────────────────────────────────────
#  OPENENV CORE TYPES
# ──────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    OpenEnv Observation — full environment state snapshot returned
    by reset() and step().
    """
    week_id: str
    task_id: str
    employees: List[Employee]
    shifts: List[Shift]
    assignments: List[Dict] = Field(default_factory=list, description="[{employee_id, shift_id, hours}]")
    unassigned_shifts: List[str] = Field(default_factory=list, description="Shift IDs with no assignment")
    conflicts: List[ConstraintViolation] = Field(default_factory=list)
    metrics: ScheduleMetrics
    done: bool = False
    step_count: int = 0
    max_steps: int = 60


class Action(BaseModel):
    """
    OpenEnv Action — describes a scheduling operation the agent wants to perform.
    action_type is always required; other fields depend on type.
    """
    action_type: ACTION_TYPES = Field(..., description="assign | remove | swap | noop")
    employee_id: Optional[str] = Field(None, description="Employee to assign/remove/swap")
    shift_id: Optional[str] = Field(None, description="Shift target for assign/remove")
    target_employee_id: Optional[str] = Field(None, description="Second employee for swap")

    class Config:
        json_schema_extra = {
            "examples": [
                {"action_type": "assign", "employee_id": "emp001", "shift_id": "shf042"},
                {"action_type": "swap", "employee_id": "emp001", "target_employee_id": "emp005"},
                {"action_type": "remove", "employee_id": "emp003", "shift_id": "shf010"},
                {"action_type": "noop"},
            ]
        }


class Reward(BaseModel):
    """
    OpenEnv Reward — dense shaped reward with component breakdown
    so agents and humans can understand why each reward was given.
    """
    total: float = Field(..., description="Scalar reward for this step, range [-1.0, 1.0]")
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Named reward components summing to total"
    )
    info: Dict[str, str] = Field(
        default_factory=dict,
        description="Human-readable explanation of what happened"
    )


class StepResult(BaseModel):
    """Full return value of step() — OpenEnv compliant."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, str] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    """Request body for POST /reset."""
    task_id: str = Field("task_medium", description="Which task to initialize: task_easy | task_medium | task_hard")
    seed: Optional[int] = Field(None, description="Random seed for reproducible episodes")


class AddEmployeeRequest(BaseModel):
    """Request body to dynamically add an employee."""
    name: str
    skills: List[str]
    max_hours_per_week: int = 40
    preferred_shift: Optional[str] = None


class EditEmployeeRequest(BaseModel):
    """Request body to edit an employee."""
    employee_id: str
    max_hours_per_week: Optional[int] = None
    preferred_shift: Optional[str] = None
    preference_weight: Optional[float] = None


class AddShiftRequest(BaseModel):
    """Request body to dynamically add a shift."""
    day: int = Field(..., ge=0, le=6)
    period: str
    duration_hours: float = 8.0
    required_skill: str
    demand_level: float = 1.0


class LeaveRequest(BaseModel):
    """Request body to apply for leave."""
    employee_id: str
    from_day: int = Field(..., ge=0, le=6)
    to_day: int = Field(..., ge=0, le=6)
    reason: Optional[str] = None


class TaskInfo(BaseModel):
    """Metadata about a single task, returned by GET /tasks."""
    id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int
    target_score: float
    action_schema: Dict = Field(..., description="JSON schema of the Action model for this task")


class GraderResult(BaseModel):
    """Result returned by GET /grader after episode completion."""
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Final normalized score 0.0–1.0")
    breakdown: Dict[str, float] = Field(..., description="Sub-scores for each grader component")
    passed: bool
    summary: str


class BaselineResult(BaseModel):
    """Result returned by POST /baseline."""
    model: str
    results: List[Dict] = Field(..., description="Per-task baseline scores")
    mean_score: float
    timestamp: str
