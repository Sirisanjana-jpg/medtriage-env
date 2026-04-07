"""
MedTriageEnv — Main environment class implementing OpenEnv interface.
step() / reset() / state() API with full typed models.
"""
from __future__ import annotations
import random
from typing import Any, Dict, Optional, Tuple
from env.models import Observation, Action, Reward, EnvironmentState
from tasks import task1_vitals, task2_drug_interactions, task3_differential_dx


TASKS = {
    "T1_vitals": task1_vitals,
    "T2_drug_interactions": task2_drug_interactions,
    "T3_differential_diagnosis": task3_differential_dx,
}

TASK_ORDER = ["T1_vitals", "T2_drug_interactions", "T3_differential_diagnosis"]


class MedTriageEnv:
    """
    OpenEnv-compliant Clinical Decision Support Environment.

    Simulates the triage workflow of an emergency department:
    1. Extract vital signs from unstructured notes (Easy)
    2. Identify drug-drug interactions (Medium)
    3. Produce ranked differential diagnosis (Hard)

    API:
        env = MedTriageEnv(task_id="T1_vitals", case_idx=0)
        obs = env.reset()
        obs, reward, done, info = env.step(action)
        state = env.state()
    """

    VERSION = "1.0.0"
    ENV_NAME = "MedTriageEnv"

    def __init__(
        self,
        task_id: str = "T1_vitals",
        case_idx: int = 0,
        seed: Optional[int] = None,
    ):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASKS.keys())}")
        self.task_id = task_id
        self.case_idx = case_idx
        self._task_module = TASKS[task_id]
        self._rng = random.Random(seed)

        self._obs: Optional[Observation] = None
        self._case: Optional[Dict] = None
        self._step: int = 0
        self._done: bool = False
        self._episode_reward: float = 0.0
        self._last_reward: Optional[Reward] = None

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._step = 0
        self._done = False
        self._episode_reward = 0.0
        self._last_reward = None
        self._obs, self._case = self._task_module.make_observation(self.case_idx, step=1)
        return self._obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one step.

        Args:
            action: Action with task_id and content (agent's response string)

        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._obs is None:
            raise RuntimeError("Call reset() before step().")

        self._step += 1
        reward = self._task_module.grade(action, self._case)
        self._episode_reward += reward.value
        self._done = reward.done

        info = {
            "task_id": self.task_id,
            "case_idx": self.case_idx,
            "step": self._step,
            "reward_breakdown": reward.breakdown,
            "feedback": reward.feedback,
        }

        self._last_reward = reward
        return self._obs, reward, self._done, info

    def state(self) -> EnvironmentState:
        """Return the current environment state."""
        return EnvironmentState(
            task_id=self.task_id,
            step=self._step,
            patient_id=self._case["patient"].patient_id if self._case else "N/A",
            episode_reward=round(self._episode_reward, 4),
            done=self._done,
            info={
                "case_idx": self.case_idx,
                "last_feedback": self._last_reward.feedback if self._last_reward else "",
            },
        )

    @classmethod
    def list_tasks(cls) -> Dict[str, str]:
        return {
            "T1_vitals": "Easy — Extract vital signs from unstructured nurse notes",
            "T2_drug_interactions": "Medium — Identify drug-drug interactions",
            "T3_differential_diagnosis": "Hard — Rank differential diagnoses for ER presentation",
        }

    @classmethod
    def num_cases(cls, task_id: str) -> int:
        mod = TASKS[task_id]
        return len(mod.CASES)


def run_all_tasks(agent_fn, seed: int = 42) -> Dict[str, Any]:
    """
    Run all tasks with all cases using agent_fn(prompt: str) -> str.
    Returns per-task average scores.
    """
    results = {}
    for task_id in TASK_ORDER:
        mod = TASKS[task_id]
        task_scores = []
        for case_idx in range(len(mod.CASES)):
            env = MedTriageEnv(task_id=task_id, case_idx=case_idx, seed=seed)
            obs = env.reset()
            prompt = obs.to_prompt()
            response = agent_fn(prompt)
            action = Action(task_id=task_id, content=response)
            _, reward, done, info = env.step(action)
            task_scores.append(reward.value)
        results[task_id] = {
            "scores": task_scores,
            "mean": round(sum(task_scores) / len(task_scores), 4),
            "task_description": mod.DESCRIPTION[:80] + "...",
        }
    return results
