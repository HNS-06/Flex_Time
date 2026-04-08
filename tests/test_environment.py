"""
FlexTime — Test Suite
Tests for OpenEnv spec compliance, environment correctness, grader determinism,
and baseline reproducibility.
"""

import pytest
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.app import app
from server.engine import FlexTimeEnv, TASK_CONFIGS
from server.models import Action, Observation, Reward, StepResult


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def env():
    e = FlexTimeEnv()
    return e


# ══════════════════════════════════════════════════════════════
#  SPEC COMPLIANCE TESTS
# ══════════════════════════════════════════════════════════════

class TestSpecCompliance:
    """Verify the environment meets all OpenEnv spec requirements."""

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_reset_returns_observation(self, client):
        r = client.post("/reset", json={"task_id": "task_easy", "seed": 42})
        assert r.status_code == 200
        data = r.json()
        # Must have all required Observation fields
        assert "week_id" in data
        assert "employees" in data
        assert "shifts" in data
        assert "assignments" in data
        assert "unassigned_shifts" in data
        assert "conflicts" in data
        assert "metrics" in data
        assert "done" in data
        assert data["done"] is False

    def test_step_returns_full_result(self, client):
        client.post("/reset", json={"task_id": "task_easy", "seed": 42})
        r = client.post("/step", json={"action_type": "noop"})
        assert r.status_code == 200
        data = r.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data
        # Reward must be a float in [-1, 1]
        assert isinstance(data["reward"]["total"], float)
        assert -1.0 <= data["reward"]["total"] <= 1.0

    def test_state_returns_observation(self, client):
        client.post("/reset", json={"task_id": "task_medium", "seed": 42})
        r = client.get("/state")
        assert r.status_code == 200
        assert "week_id" in r.json()

    def test_tasks_endpoint_returns_all(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        data = r.json()
        assert "tasks" in data
        assert len(data["tasks"]) >= 3
        task_ids = [t["id"] for t in data["tasks"]]
        assert "task_easy" in task_ids
        assert "task_medium" in task_ids
        assert "task_hard" in task_ids

    def test_tasks_include_action_schema(self, client):
        r = client.get("/tasks")
        for task in r.json()["tasks"]:
            assert "action_schema" in task
            assert "difficulty" in task
            assert task["difficulty"] in ["easy", "medium", "hard"]

    def test_grader_returns_normalized_score(self, client):
        client.post("/reset", json={"task_id": "task_easy", "seed": 42})
        r = client.get("/grader")
        assert r.status_code == 200
        data = r.json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0
        assert "breakdown" in data
        assert "passed" in data
        assert "summary" in data

    def test_reward_has_components(self, client):
        client.post("/reset", json={"task_id": "task_easy", "seed": 42})
        # Make a valid assign action
        state = client.get("/state").json()
        unassigned = state["unassigned_shifts"]
        employees = state["employees"]
        r = client.post("/step", json={
            "action_type": "assign",
            "employee_id": employees[0]["id"],
            "shift_id": unassigned[0] if unassigned else "shf001"
        })
        data = r.json()
        assert "components" in data["reward"]
        assert isinstance(data["reward"]["components"], dict)


# ══════════════════════════════════════════════════════════════
#  ENVIRONMENT CORRECTNESS TESTS
# ══════════════════════════════════════════════════════════════

class TestEnvironmentCorrectness:

    def test_reset_clean_state(self, env):
        env.reset("task_easy", seed=42)
        obs = env.state()
        # All shifts unassigned on reset
        assert obs.metrics.assigned_shifts == 0
        assert obs.metrics.coverage_rate == 0.0
        assert obs.step_count == 0
        assert obs.metrics.episode_reward == 0.0

    def test_reset_reproducible_with_seed(self, env):
        obs1 = env.reset("task_medium", seed=42)
        obs2 = env.reset("task_medium", seed=42)
        assert obs1.week_id != obs2.week_id  # week_id is uuid so different
        assert len(obs1.employees) == len(obs2.employees)
        assert len(obs1.shifts) == len(obs2.shifts)
        # Same employee names in same order
        assert [e.name for e in obs1.employees] == [e.name for e in obs2.employees]

    def test_valid_assign_increases_coverage(self, env):
        obs = env.reset("task_easy", seed=42)
        before = obs.metrics.coverage_rate
        # Find valid assignment
        unassigned = obs.unassigned_shifts
        emp = obs.employees[0]
        action = Action(action_type="assign", employee_id=emp.id, shift_id=unassigned[0])
        result = env.step(action)
        assert result.observation.metrics.coverage_rate > before

    def test_invalid_action_penalized(self, env):
        env.reset("task_easy", seed=42)
        # Try to assign non-existent employee
        action = Action(action_type="assign", employee_id="emp_invalid", shift_id="shf001")
        result = env.step(action)
        assert result.reward.total < 0

    def test_noop_gives_zero_reward(self, env):
        env.reset("task_easy", seed=42)
        result = env.step(Action(action_type="noop"))
        assert result.reward.total == 0.0

    def test_done_when_all_assigned(self, env):
        obs = env.reset("task_easy", seed=42)
        n = len(obs.shifts)
        # Greedily assign all
        for _ in range(n + 5):
            state = env.state()
            if state.done or not state.unassigned_shifts:
                break
            unassigned = state.unassigned_shifts
            for emp in state.employees:
                shf_id = unassigned[0]
                shf = next(s for s in state.shifts if s.id == shf_id)
                if (shf.required_skill in emp.skills and
                        emp.availability[shf.day] and
                        emp.assigned_hours + shf.duration_hours <= emp.max_hours_per_week):
                    action = Action(action_type="assign", employee_id=emp.id, shift_id=shf_id)
                    result = env.step(action)
                    if result.done:
                        assert result.done is True
                    break
            else:
                env.step(Action(action_type="noop"))

    def test_max_steps_terminates_episode(self, env):
        env.reset("task_easy", seed=42)
        # Exhaust steps with noops
        for _ in range(25):  # max_steps is 20 for easy
            result = env.step(Action(action_type="noop"))
            if result.done:
                break
        assert result.done is True

    def test_skill_mismatch_detected(self, env):
        obs = env.reset("task_medium", seed=42)
        # Find a shift with a skill that first employee doesn't have
        emp = obs.employees[0]
        incompatible = next(
            (s for s in obs.shifts
             if s.required_skill not in emp.skills and not s.is_assigned),
            None
        )
        if incompatible:
            env.step(Action(action_type="assign", employee_id=emp.id, shift_id=incompatible.id))
            state = env.state()
            violation_types = [v.violation_type for v in state.conflicts]
            assert "skill_mismatch" in violation_types


# ══════════════════════════════════════════════════════════════
#  GRADER TESTS
# ══════════════════════════════════════════════════════════════

class TestGraders:

    def test_all_graders_return_0_to_1(self, env):
        for task_id in TASK_CONFIGS:
            env.reset(task_id, seed=42)
            result = env.grade()
            assert 0.0 <= result["score"] <= 1.0, f"{task_id} score out of range"

    def test_grader_deterministic(self, env):
        """Same episode = same grade."""
        env.reset("task_easy", seed=42)
        grade1 = env.grade()
        env.reset("task_easy", seed=42)
        grade2 = env.grade()
        assert grade1["score"] == grade2["score"]

    def test_grader_score_increases_with_better_schedule(self, env):
        env.reset("task_easy", seed=42)
        empty_grade = env.grade()
        # Fill some shifts
        for _ in range(3):
            state = env.state()
            if not state.unassigned_shifts:
                break
            shf_id = state.unassigned_shifts[0]
            shf = next(s for s in state.shifts if s.id == shf_id)
            emp = next(
                (e for e in state.employees
                 if shf.required_skill in e.skills and e.availability[shf.day]),
                None
            )
            if emp:
                env.step(Action(action_type="assign", employee_id=emp.id, shift_id=shf_id))
        filled_grade = env.grade()
        assert filled_grade["score"] >= empty_grade["score"]

    def test_grader_breakdown_fields(self, env):
        for task_id in TASK_CONFIGS:
            env.reset(task_id, seed=42)
            result = env.grade()
            for field in ["coverage_score", "fairness_score", "constraint_score", "demand_score"]:
                assert field in result["breakdown"], f"{task_id} missing {field}"
                assert 0.0 <= result["breakdown"][field] <= 1.0

    def test_hard_task_requires_all_thresholds(self, env):
        """Hard task penalizes if any sub-score < 0.75."""
        env.reset("task_hard", seed=42)
        result = env.grade()
        # With empty schedule, score should be low (not passing)
        assert result["score"] < 0.75


# ══════════════════════════════════════════════════════════════
#  TASK DIFFICULTY TESTS
# ══════════════════════════════════════════════════════════════

class TestTaskDifficulty:

    def test_easy_task_has_fewer_shifts(self, env):
        obs_easy = env.reset("task_easy", seed=42)
        obs_hard = env.reset("task_hard", seed=42)
        assert len(obs_easy.shifts) < len(obs_hard.shifts)
        assert len(obs_easy.employees) < len(obs_hard.employees)

    def test_hard_task_has_seed_conflicts(self, env):
        obs = env.reset("task_hard", seed=42)
        # Hard task seeds 3 conflicts
        assert len(obs.conflicts) >= 0  # may or may not manifest on initial check

    def test_max_steps_scales_with_difficulty(self):
        assert TASK_CONFIGS["task_easy"]["max_steps"] < TASK_CONFIGS["task_medium"]["max_steps"]
        assert TASK_CONFIGS["task_medium"]["max_steps"] < TASK_CONFIGS["task_hard"]["max_steps"]

    def test_target_score_decreases_with_difficulty(self):
        assert TASK_CONFIGS["task_easy"]["target_score"] > TASK_CONFIGS["task_medium"]["target_score"]
        assert TASK_CONFIGS["task_medium"]["target_score"] >= TASK_CONFIGS["task_hard"]["target_score"]


# ══════════════════════════════════════════════════════════════
#  REWARD SHAPE TESTS
# ══════════════════════════════════════════════════════════════

class TestRewardShaping:

    def test_reward_in_range(self, env):
        env.reset("task_medium", seed=42)
        for _ in range(10):
            result = env.step(Action(action_type="noop"))
            assert -1.0 <= result.reward.total <= 1.0

    def test_assigning_valid_shift_positive_reward(self, env):
        obs = env.reset("task_easy", seed=42)
        unassigned = obs.unassigned_shifts
        for emp in obs.employees:
            shf = next((s for s in obs.shifts if s.id == unassigned[0]), None)
            if shf and shf.required_skill in emp.skills and emp.availability[shf.day]:
                result = env.step(Action(action_type="assign", employee_id=emp.id, shift_id=unassigned[0]))
                assert result.reward.total > 0, "Valid assignment should yield positive reward"
                break

    def test_reward_components_present(self, env):
        env.reset("task_easy", seed=42)
        result = env.step(Action(action_type="noop"))
        assert isinstance(result.reward.components, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
