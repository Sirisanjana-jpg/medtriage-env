"""
MedTriageEnv — Typed Pydantic models for OpenEnv compliance.
Observation, Action, Reward, and State types.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time


class PatientRecord(BaseModel):
    patient_id: str
    age: int
    sex: str  # "M" | "F" | "other"
    chief_complaint: str
    nurse_notes: str
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    vitals_raw: Optional[str] = None   # unstructured text (Task 1)
    history: Optional[str] = None


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    task_description: str
    patient: PatientRecord
    step: int
    max_steps: int
    context: Dict[str, Any] = Field(default_factory=dict)

    def to_prompt(self) -> str:
        p = self.patient
        lines = [
            f"=== TASK: {self.task_description} ===",
            f"Patient ID: {p.patient_id} | Age: {p.age} | Sex: {p.sex}",
            f"Chief complaint: {p.chief_complaint}",
        ]
        if p.nurse_notes:
            lines.append(f"\nNurse notes:\n{p.nurse_notes}")
        if p.medications:
            lines.append(f"\nCurrent medications: {', '.join(p.medications)}")
        if p.allergies:
            lines.append(f"Allergies: {', '.join(p.allergies)}")
        if p.history:
            lines.append(f"\nMedical history: {p.history}")
        if self.context:
            for k, v in self.context.items():
                lines.append(f"\n{k}: {v}")
        lines.append(f"\nStep {self.step}/{self.max_steps}")
        return "\n".join(lines)


class Action(BaseModel):
    """What the agent returns."""
    task_id: str
    content: str                        # free-form agent response
    structured: Dict[str, Any] = Field(default_factory=dict)  # parsed fields


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = ""
    done: bool = False


class EnvironmentState(BaseModel):
    task_id: str
    step: int
    patient_id: str
    episode_reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
