"""
FlexTime — Environment Engine
Core scheduling simulation: state management, constraint checking,
reward computation, and episode lifecycle.
"""

from __future__ import annotations
import random
import uuid
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from server.models import (
    Action, ConstraintViolation, Employee, Observation,
    Reward, ScheduleMetrics, Shift, StepResult,
)


# ──────────────────────────────────────────────────────────────
#  SCENARIO FACTORIES
# ──────────────────────────────────────────────────────────────

SKILL_POOL = ["cashier", "supervisor", "inventory", "customer_service", "technician"]

EMPLOYEE_TEMPLATES = [
    ("Alice K.",   ["cashier", "supervisor"],        [1,1,1,1,1,0,0], 40, "morning",   1.2),
    ("Bob M.",     ["cashier", "inventory"],         [1,1,0,1,1,1,0], 40, "afternoon", 1.0),
    ("Carol T.",   ["cashier"],                      [1,1,1,0,1,0,0], 32, "morning",   1.1),
    ("David R.",   ["supervisor", "inventory"],      [0,1,1,1,1,0,1], 40, "night",     1.0),
    ("Emma S.",    ["cashier", "customer_service"],  [1,0,1,1,0,1,1], 40, "morning",   1.3),
    ("Frank L.",   ["supervisor"],                   [1,1,1,1,1,1,0], 40, "afternoon", 0.9),
    ("Grace H.",   ["cashier", "customer_service"],  [1,1,0,0,1,1,1], 40, "morning",   1.0),
    ("Henry P.",   ["inventory"],                    [0,1,1,1,1,0,0], 40, "afternoon", 1.0),
    ("Iris W.",    ["cashier", "technician"],        [1,0,0,1,1,1,0], 40, "night",     1.1),
    ("James O.",   ["supervisor", "customer_service"],[1,1,1,0,0,1,1], 40, "morning",  1.2),
    ("Karen B.",   ["cashier"],                      [0,1,1,1,0,1,1], 32, "afternoon", 1.0),
    ("Leo M.",     ["technician", "inventory"],      [1,1,0,1,1,0,0], 40, "night",     0.8),
]


def _make_employees(n: int, rng: random.Random) -> List[Employee]:
    templates = EMPLOYEE_TEMPLATES[:n]
    employees = []
    for i, (name, skills, avail, max_h, pref, weight) in enumerate(templates):
        employees.append(Employee(
            id=f"emp{i+1:03d}",
            name=name,
            skills=skills,
            availability=avail,
            max_hours_per_week=max_h,
            assigned_hours=0.0,
            preferred_shift=pref,
            preference_weight=weight,
        ))
    return employees


def _make_shifts(
    n: int,
    required_skills: List[str],
    rng: random.Random,
    seed_conflicts: int = 0,
) -> List[Shift]:
    periods = ["morning", "afternoon", "night"]
    shifts = []
    for i in range(n):
        day = i % 7
        period = periods[i % 3]
        skill = required_skills[i % len(required_skills)]
        demand = round(rng.uniform(0.8, 2.5), 2)
        shifts.append(Shift(
            id=f"shf{i+1:03d}",
            day=day,
            period=period,
            duration_hours=8.0,
            required_skill=skill,
            demand_level=demand,
            assigned_employee_id=None,
        ))
    # Seed some pre-assigned conflicts for the hard task
    for j in range(seed_conflicts):
        shifts[j].assigned_employee_id = f"emp{j+1:03d}"  # will create overlaps
    return shifts


# ──────────────────────────────────────────────────────────────
#  TASK SCENARIOS
# ──────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "task_easy": {
        "name": "Basic Shift Coverage",
        "difficulty": "easy",
        "description": (
            "Assign employees to all 5 open morning shifts for a single day. "
            "All employees are available; skills match every shift. The agent "
            "simply needs to fill all slots without overlap."
        ),
        "n_employees": 5,
        "n_shifts": 5,
        "required_skills": ["cashier"],
        "max_steps": 20,
        "target_score": 1.0,
        "seed_conflicts": 0,
    },
    "task_medium": {
        "name": "Weekly Schedule with Constraints",
        "difficulty": "medium",
        "description": (
            "Build a complete weekly schedule for 8 employees across 30 shifts, "
            "respecting skill requirements, availability windows, and the 40-hour "
            "maximum working week. Partial coverage is scored proportionally."
        ),
        "n_employees": 8,
        "n_shifts": 30,
        "required_skills": ["cashier", "supervisor", "inventory"],
        "max_steps": 60,
        "target_score": 0.85,
        "seed_conflicts": 0,
    },
    "task_hard": {
        "name": "Fair Optimization Under Pressure",
        "difficulty": "hard",
        "description": (
            "Generate an optimal weekly schedule for 12 employees and 50 shifts "
            "maximizing coverage while minimizing the fairness delta (≤4h spread), "
            "resolving 3 pre-seeded conflicts, and satisfying employee preference "
            "weights. Must score ≥ 0.75 on all sub-metrics simultaneously."
        ),
        "n_employees": 12,
        "n_shifts": 50,
        "required_skills": ["cashier", "supervisor", "inventory", "customer_service", "technician"],
        "max_steps": 120,
        "target_score": 0.75,
        "seed_conflicts": 3,
    },
}


# ──────────────────────────────────────────────────────────────
#  CONSTRAINT CHECKER
# ──────────────────────────────────────────────────────────────

class ConstraintChecker:
    def check_all(
        self,
        employees: List[Employee],
        shifts: List[Shift],
    ) -> List[ConstraintViolation]:
        violations: List[ConstraintViolation] = []
        emp_map = {e.id: e for e in employees}
        shift_map = {s.id: s for s in shifts}

        # Group shifts per employee
        emp_shifts: Dict[str, List[Shift]] = {e.id: [] for e in employees}
        for s in shifts:
            if s.assigned_employee_id and s.assigned_employee_id in emp_shifts:
                emp_shifts[s.assigned_employee_id].append(s)

        for emp_id, emp_shift_list in emp_shifts.items():
            emp = emp_map.get(emp_id)
            if not emp:
                continue

            # H1: No overlapping shifts (same day, same period)
            seen_slots = set()
            for s in emp_shift_list:
                slot = (s.day, s.period)
                if slot in seen_slots:
                    violations.append(ConstraintViolation(
                        violation_type="overlap",
                        employee_id=emp_id,
                        shift_id=s.id,
                        description=f"{emp.name} has overlapping shifts on {s.day_name} {s.period}",
                        severity="hard",
                        penalty=0.20,
                    ))
                seen_slots.add(slot)

            # H2: Skill match
            for s in emp_shift_list:
                if s.required_skill not in emp.skills:
                    violations.append(ConstraintViolation(
                        violation_type="skill_mismatch",
                        employee_id=emp_id,
                        shift_id=s.id,
                        description=f"{emp.name} lacks skill '{s.required_skill}' for shift {s.id}",
                        severity="hard",
                        penalty=0.20,
                    ))

            # H3: Availability
            for s in emp_shift_list:
                if not emp.availability[s.day]:
                    violations.append(ConstraintViolation(
                        violation_type="unavailable",
                        employee_id=emp_id,
                        shift_id=s.id,
                        description=f"{emp.name} is unavailable on {s.day_name}",
                        severity="hard",
                        penalty=0.20,
                    ))

            # H4: Max hours
            if emp.is_overloaded:
                violations.append(ConstraintViolation(
                    violation_type="max_hours",
                    employee_id=emp_id,
                    shift_id=None,
                    description=f"{emp.name} exceeds max hours ({emp.assigned_hours:.0f}h > {emp.max_hours_per_week}h)",
                    severity="hard",
                    penalty=0.10,
                ))

        # S1: Fairness — check max-min hours spread
        all_hours = [e.assigned_hours for e in employees if any(
            s.assigned_employee_id == e.id for s in shifts
        )]
        if len(all_hours) > 1:
            delta = max(all_hours) - min(all_hours)
            if delta > 4.0:
                violations.append(ConstraintViolation(
                    violation_type="fairness",
                    employee_id=None,
                    shift_id=None,
                    description=f"Hour imbalance {delta:.1f}h exceeds 4h threshold",
                    severity="soft",
                    penalty=0.05 * min(delta / 4.0, 3.0),
                ))

        return violations


# ──────────────────────────────────────────────────────────────
#  REWARD SHAPER
# ──────────────────────────────────────────────────────────────

class RewardShaper:
    def compute(
        self,
        action: Action,
        prev_metrics: ScheduleMetrics,
        new_metrics: ScheduleMetrics,
        new_violations: List[ConstraintViolation],
        prev_violations: List[ConstraintViolation],
        action_valid: bool,
        newly_assigned: bool,
        newly_removed: bool,
        conflict_resolved: bool,
    ) -> Reward:
        components: Dict[str, float] = {}

        if not action_valid:
            components["invalid_action"] = -0.05
            return Reward(
                total=-0.05,
                components=components,
                info={"result": "invalid action — no state change"},
            )

        if action.action_type == "noop":
            components["noop"] = 0.0
            return Reward(total=0.0, components=components, info={"result": "no-op"})

        # Coverage improvement
        cov_delta = new_metrics.coverage_rate - prev_metrics.coverage_rate
        if cov_delta > 0:
            components["shift_covered"] = round(cov_delta * 1.5, 4)
        elif cov_delta < 0 and newly_removed:
            components["shift_uncovered"] = round(cov_delta * 1.0, 4)

        # Demand-weighted coverage
        demand_delta = new_metrics.demand_satisfaction - prev_metrics.demand_satisfaction
        components["demand_signal"] = round(demand_delta * 0.5, 4)

        # Hard violations
        new_hard = sum(1 for v in new_violations if v.severity == "hard")
        prev_hard = sum(1 for v in prev_violations if v.severity == "hard")
        viol_delta = new_hard - prev_hard
        if viol_delta > 0:
            components["constraint_violated"] = round(-0.20 * viol_delta, 4)
        elif viol_delta < 0:
            components["constraint_resolved"] = round(0.10 * abs(viol_delta), 4)

        # Conflict resolution bonus
        if conflict_resolved:
            components["conflict_resolved"] = components.get("conflict_resolved", 0) + 0.10

        # Fairness improvement
        fair_delta = new_metrics.fairness_score - prev_metrics.fairness_score
        components["fairness"] = round(fair_delta * 0.3, 4)

        # Preference satisfaction
        pref_delta = new_metrics.preference_satisfaction - prev_metrics.preference_satisfaction
        components["preference"] = round(pref_delta * 0.1, 4)

        total = sum(components.values())
        total = max(-1.0, min(1.0, total))

        info = {"result": f"action={action.action_type}, Δcoverage={cov_delta:+.3f}, hard_violations={new_hard}"}
        return Reward(total=round(total, 4), components=components, info=info)


# ──────────────────────────────────────────────────────────────
#  ENVIRONMENT ENGINE
# ──────────────────────────────────────────────────────────────

class FlexTimeEnv:
    """
    Core FlexTime environment engine.
    Manages episode lifecycle, state transitions, and reward computation.
    """

    def __init__(self) -> None:
        self._task_id: str = "task_medium"
        self._employees: List[Employee] = []
        self._shifts: List[Shift] = []
        self._step_count: int = 0
        self._max_steps: int = 60
        self._episode_reward: float = 0.0
        self._week_id: str = ""
        self._checker = ConstraintChecker()
        self._shaper = RewardShaper()
        self._violations: List[ConstraintViolation] = []
        self._rng = random.Random(42)
        self._initialized: bool = False

    # ── reset ──────────────────────────────────────────────────
    def reset(self, task_id: str = "task_medium", seed: Optional[int] = None) -> Observation:
        cfg = TASK_CONFIGS.get(task_id)
        if not cfg:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASK_CONFIGS)}")

        self._task_id = task_id
        self._max_steps = cfg["max_steps"]
        self._step_count = 0
        self._episode_reward = 0.0
        self._week_id = f"week-{uuid.uuid4().hex[:6]}"
        self._rng = random.Random(seed if seed is not None else 42)

        self._employees = _make_employees(cfg["n_employees"], self._rng)
        self._shifts = _make_shifts(
            cfg["n_shifts"],
            cfg["required_skills"],
            self._rng,
            cfg["seed_conflicts"],
        )
        self._violations = self._checker.check_all(self._employees, self._shifts)
        self._initialized = True

        return self._build_observation(done=False)

    # ── step ───────────────────────────────────────────────────
    def step(self, action: Action) -> StepResult:
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        prev_metrics = self._build_metrics()
        prev_violations = list(self._violations)

        valid, conflict_resolved, newly_assigned, newly_removed = self._apply_action(action)

        self._violations = self._checker.check_all(self._employees, self._shifts)
        new_metrics = self._build_metrics()

        reward = self._shaper.compute(
            action=action,
            prev_metrics=prev_metrics,
            new_metrics=new_metrics,
            new_violations=self._violations,
            prev_violations=prev_violations,
            action_valid=valid,
            newly_assigned=newly_assigned,
            newly_removed=newly_removed,
            conflict_resolved=conflict_resolved,
        )

        self._step_count += 1
        self._episode_reward += reward.total

        all_assigned = all(s.is_assigned for s in self._shifts)
        done = all_assigned or self._step_count >= self._max_steps

        obs = self._build_observation(done=done)
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "step": str(self._step_count),
                "episode_reward": f"{self._episode_reward:.4f}",
                "coverage": f"{new_metrics.coverage_rate:.3f}",
                "hard_violations": str(new_metrics.hard_violations),
            },
        )

    # ── state ──────────────────────────────────────────────────
    def state(self) -> Observation:
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._build_observation(done=False)

    # ── action application ─────────────────────────────────────
    def _apply_action(
        self, action: Action
    ) -> Tuple[bool, bool, bool, bool]:
        """Returns (valid, conflict_resolved, newly_assigned, newly_removed)."""

        emp_map = {e.id: e for e in self._employees}
        shift_map = {s.id: s for s in self._shifts}

        if action.action_type == "noop":
            return True, False, False, False

        if action.action_type == "assign":
            emp = emp_map.get(action.employee_id or "")
            shf = shift_map.get(action.shift_id or "")
            if not emp or not shf:
                return False, False, False, False
            if shf.is_assigned:
                return False, False, False, False  # already filled
            if action.employee_id not in [e.id for e in self._employees]:
                return False, False, False, False

            prev_hard = sum(1 for v in self._violations if v.severity == "hard")
            shf.assigned_employee_id = emp.id
            emp.assigned_hours += shf.duration_hours
            new_viols = self._checker.check_all(self._employees, self._shifts)
            new_hard = sum(1 for v in new_viols if v.severity == "hard")
            conflict_resolved = new_hard < prev_hard
            return True, conflict_resolved, True, False

        if action.action_type == "remove":
            emp = emp_map.get(action.employee_id or "")
            shf = shift_map.get(action.shift_id or "")
            if not emp or not shf:
                return False, False, False, False
            if shf.assigned_employee_id != emp.id:
                return False, False, False, False

            shf.assigned_employee_id = None
            emp.assigned_hours = max(0.0, emp.assigned_hours - shf.duration_hours)
            return True, False, False, True

        if action.action_type == "swap":
            emp_a = emp_map.get(action.employee_id or "")
            emp_b = emp_map.get(action.target_employee_id or "")
            if not emp_a or not emp_b:
                return False, False, False, False

            prev_hard = sum(1 for v in self._violations if v.severity == "hard")
            # Swap all assignments between the two employees
            for shf in self._shifts:
                if shf.assigned_employee_id == emp_a.id:
                    shf.assigned_employee_id = emp_b.id
                elif shf.assigned_employee_id == emp_b.id:
                    shf.assigned_employee_id = emp_a.id

            # Recompute hours
            for emp in [emp_a, emp_b]:
                emp.assigned_hours = sum(
                    s.duration_hours for s in self._shifts if s.assigned_employee_id == emp.id
                )

            new_viols = self._checker.check_all(self._employees, self._shifts)
            new_hard = sum(1 for v in new_viols if v.severity == "hard")
            conflict_resolved = new_hard < prev_hard
            return True, conflict_resolved, False, False

        return False, False, False, False

    # ── metrics ────────────────────────────────────────────────
    def _build_metrics(self) -> ScheduleMetrics:
        total = len(self._shifts)
        assigned = sum(1 for s in self._shifts if s.is_assigned)
        coverage = assigned / total if total > 0 else 0.0

        hard_v = sum(1 for v in self._violations if v.severity == "hard")
        soft_v = sum(1 for v in self._violations if v.severity == "soft")

        # Demand-weighted satisfaction
        total_demand = sum(s.demand_level for s in self._shifts)
        met_demand = sum(s.demand_level for s in self._shifts if s.is_assigned)
        demand_sat = met_demand / total_demand if total_demand > 0 else 0.0

        # Fairness
        active_hours = [e.assigned_hours for e in self._employees if e.assigned_hours > 0]
        if len(active_hours) >= 2:
            delta = max(active_hours) - min(active_hours)
        else:
            delta = 0.0
        fairness_score = max(0.0, 1.0 - delta / 40.0)

        # Preference satisfaction
        pref_scores = []
        for s in self._shifts:
            if not s.is_assigned:
                continue
            emp = next((e for e in self._employees if e.id == s.assigned_employee_id), None)
            if emp and emp.preferred_shift:
                match = 1.0 if emp.preferred_shift == s.period else 0.3
                pref_scores.append(match * emp.preference_weight)
        pref_sat = sum(pref_scores) / max(1, len(pref_scores)) if pref_scores else 0.0
        pref_sat = min(1.0, pref_sat)

        # Avg hours and unmet demand
        avg_hours = sum(e.assigned_hours for e in self._employees) / max(1, len(self._employees))
        unmet_demand = total - assigned

        return ScheduleMetrics(
            total_shifts=total,
            assigned_shifts=assigned,
            coverage_rate=round(coverage, 4),
            hard_violations=hard_v,
            soft_violations=soft_v,
            fairness_delta=round(delta, 2),
            fairness_score=round(fairness_score, 4),
            demand_satisfaction=round(demand_sat, 4),
            preference_satisfaction=round(pref_sat, 4),
            avg_hours=round(avg_hours, 2),
            unmet_demand=unmet_demand,
            step_count=self._step_count,
            episode_reward=round(self._episode_reward, 4),
        )

    # ── observation builder ────────────────────────────────────
    def _build_observation(self, done: bool) -> Observation:
        metrics = self._build_metrics()
        assignments = [
            {"employee_id": s.assigned_employee_id, "shift_id": s.id, "hours": s.duration_hours}
            for s in self._shifts if s.is_assigned
        ]
        unassigned = [s.id for s in self._shifts if not s.is_assigned]

        return Observation(
            week_id=self._week_id,
            task_id=self._task_id,
            employees=deepcopy(self._employees),
            shifts=deepcopy(self._shifts),
            assignments=assignments,
            unassigned_shifts=unassigned,
            conflicts=list(self._violations),
            metrics=metrics,
            done=done,
            step_count=self._step_count,
            max_steps=self._max_steps,
        )

    # ── dynamic mutators (SaaS UI enhancements) ────────────────
    def add_employee(self, emp_data: dict) -> Observation:
        """Dynamically add an employee mid-episode."""
        new_emp = Employee(
            id=f"emp{len(self._employees)+1:03d}_{str(uuid.uuid4())[:4]}",
            availability=[1]*7,
            **emp_data
        )
        self._employees.append(new_emp)
        self._reevaluate_state()
        return self.state()

    def edit_employee(self, emp_id: str, edits: dict) -> Observation:
        """Edit an existing employee."""
        emp = next((e for e in self._employees if e.id == emp_id), None)
        if not emp:
            raise ValueError(f"Employee {emp_id} not found.")
        if "max_hours_per_week" in edits and edits["max_hours_per_week"] is not None:
            emp.max_hours_per_week = edits["max_hours_per_week"]
        if "preferred_shift" in edits and edits["preferred_shift"] is not None:
            emp.preferred_shift = edits["preferred_shift"]
        if "preference_weight" in edits and edits["preference_weight"] is not None:
            emp.preference_weight = edits["preference_weight"]
            
        self._reevaluate_state()
        return self.state()

    def add_shift(self, shift_data: dict) -> Observation:
        """Dynamically add a shift mid-episode."""
        new_shift = Shift(
            id=f"shf{len(self._shifts)+1:03d}_{str(uuid.uuid4())[:4]}",
            assigned_employee_id=None,
            **shift_data
        )
        self._shifts.append(new_shift)
        self._reevaluate_state()
        return self.state()

    def apply_leave(self, emp_id: str, day_start: int, day_end: int) -> Observation:
        """Process leave request: clear availability and drop assigned shifts."""
        emp = next((e for e in self._employees if e.id == emp_id), None)
        if not emp:
            raise ValueError(f"Employee {emp_id} not found.")
            
        # Set availability to 0 for leave days
        for day in range(day_start, day_end + 1):
            if 0 <= day <= 6:
                emp.availability[day] = 0
                
        # Drop conflicting shifts
        hours_dropped = 0.0
        for s in self._shifts:
            if s.assigned_employee_id == emp_id and day_start <= s.day <= day_end:
                s.assigned_employee_id = None
                hours_dropped += s.duration_hours
                
        emp.assigned_hours = max(0.0, emp.assigned_hours - hours_dropped)
        self._reevaluate_state()
        return self.state()

    def apply_scenario(self, scenario_type: str) -> Observation:
        """Mutate environment state based on scenario template."""
        if scenario_type == "shortage":
            # Drop availability of ~30% of employees completely (simulated no-show)
            num_drop = max(1, int(len(self._employees) * 0.3))
            drop_candidates = random.sample(self._employees, num_drop)
            for emp in drop_candidates:
                emp.availability = [0] * 7
                # drop shifts
                for s in self._shifts:
                    if s.assigned_employee_id == emp.id:
                        s.assigned_employee_id = None
                        emp.assigned_hours = max(0.0, emp.assigned_hours - s.duration_hours)
                        
        elif scenario_type == "surge":
            # Add 20% more high priority shifts dynamically
            num_surge = max(2, int(len(self._shifts) * 0.2))
            skills = list({s.required_skill for s in self._shifts}) or ["cashier", "nurse", "support"]
            periods = ["morning", "afternoon", "night"]
            for i in range(num_surge):
                self._shifts.append(Shift(
                    id=f"surge_shf_{str(uuid.uuid4())[:4]}",
                    day=random.randint(0, 6),
                    period=random.choice(periods),
                    duration_hours=8.0,
                    required_skill=random.choice(skills),
                    demand_level=3.0,  # Peak demand
                    assigned_employee_id=None
                ))
                
        elif scenario_type == "holiday":
            # 25% of staff goes on leave for the final 3 days
            num_leave = max(1, int(len(self._employees) * 0.25))
            for emp in random.sample(self._employees, num_leave):
                self.apply_leave(emp.id, 4, 6) # Fri, Sat, Sun
            # Demand is also multiplied
            for s in self._shifts:
                s.demand_level = min(3.0, s.demand_level * 1.5)

        self._reevaluate_state()
        return self.state()

    # ── grader ─────────────────────────────────────────────────
    def grade(self) -> Dict:
        """Compute final normalized score for the completed episode."""
        metrics = self._build_metrics()
        cfg = TASK_CONFIGS[self._task_id]
        hard_v = metrics.hard_violations

        # Sub-scores
        coverage_score = metrics.coverage_rate
        fairness_score = metrics.fairness_score
        constraint_score = max(0.0, 1.0 - hard_v * 0.10)
        demand_score = metrics.demand_satisfaction
        pref_score = metrics.preference_satisfaction

        if self._task_id == "task_easy":
            # Weighted: coverage dominates
            final = (
                0.70 * coverage_score +
                0.15 * constraint_score +
                0.10 * fairness_score +
                0.05 * pref_score
            )
        elif self._task_id == "task_medium":
            final = (
                0.40 * coverage_score +
                0.25 * constraint_score +
                0.20 * fairness_score +
                0.10 * demand_score +
                0.05 * pref_score
            )
        else:  # task_hard — all sub-scores must be ≥ 0.75
            sub_scores = [coverage_score, fairness_score, constraint_score, demand_score]
            if all(s >= 0.75 for s in sub_scores):
                # Bonus for meeting all thresholds simultaneously
                final = (
                    0.30 * coverage_score +
                    0.25 * fairness_score +
                    0.20 * constraint_score +
                    0.15 * demand_score +
                    0.10 * pref_score
                )
            else:
                # Penalty for missing any threshold
                final = min(sub_scores) * 0.9

        final = round(max(0.0, min(1.0, final)), 4)
        passed = final >= cfg["target_score"]

        return {
            "task_id": self._task_id,
            "score": final,
            "breakdown": {
                "coverage_score": round(coverage_score, 4),
                "fairness_score": round(fairness_score, 4),
                "constraint_score": round(constraint_score, 4),
                "demand_score": round(demand_score, 4),
                "preference_score": round(pref_score, 4),
            },
            "passed": passed,
            "summary": (
                f"Score {final:.4f} ({'PASS' if passed else 'FAIL'}) — "
                f"Coverage {coverage_score:.1%}, Fairness {fairness_score:.1%}, "
                f"Hard violations: {hard_v}, Steps used: {self._step_count}/{self._max_steps}"
            ),
        }
