"""
Task 1 (Easy): Vital Signs Extraction
The agent receives unstructured nurse notes and must extract structured vitals.
Grader uses token-level F1 on each vital field.
"""
from __future__ import annotations
import re
import json
from typing import Any, Dict, List, Tuple
from env.models import PatientRecord, Observation, Action, Reward


TASK_ID = "T1_vitals"
DESCRIPTION = (
    "Extract structured vital signs from the nurse's unstructured admission notes. "
    "Return a JSON object with keys: heart_rate (bpm int), systolic_bp (mmHg int), "
    "diastolic_bp (mmHg int), spo2 (% int), temperature (°C float), respiratory_rate (breaths/min int). "
    "Use null for any vital that is not mentioned."
)

CASES: List[Dict[str, Any]] = [
    {
        "patient": PatientRecord(
            patient_id="P001",
            age=67,
            sex="F",
            chief_complaint="Shortness of breath",
            nurse_notes=(
                "Pt arrived via ambulance at 14:32. Appears in moderate distress. "
                "HR 112, BP 145/92, SpO2 88% on room air, temp 38.4C, RR 24. "
                "Reports worsening SOB over 3 days. Productive cough with yellow sputum."
            ),
            vitals_raw="HR 112, BP 145/92, SpO2 88% on room air, temp 38.4C, RR 24",
        ),
        "ground_truth": {
            "heart_rate": 112,
            "systolic_bp": 145,
            "diastolic_bp": 92,
            "spo2": 88,
            "temperature": 38.4,
            "respiratory_rate": 24,
        },
    },
    {
        "patient": PatientRecord(
            patient_id="P002",
            age=34,
            sex="M",
            chief_complaint="Chest pain",
            nurse_notes=(
                "32yo M presents with sudden onset chest pain, radiating to left arm. "
                "Vitals: pulse 98 bpm, blood pressure 138 over 85, O2 sat 97%, "
                "temp 36.9 degrees Celsius, breathing 18 times per minute."
            ),
            vitals_raw="pulse 98 bpm, blood pressure 138 over 85, O2 sat 97%, temp 36.9 degrees Celsius, breathing 18 times per minute",
        ),
        "ground_truth": {
            "heart_rate": 98,
            "systolic_bp": 138,
            "diastolic_bp": 85,
            "spo2": 97,
            "temperature": 36.9,
            "respiratory_rate": 18,
        },
    },
    {
        "patient": PatientRecord(
            patient_id="P003",
            age=78,
            sex="F",
            chief_complaint="Fall with hip pain",
            nurse_notes=(
                "Elderly female brought by family after fall at home. Alert and oriented x3. "
                "HR=72, BP=118/76mmHg, sat 94%, T 37.1C, RR 16. "
                "Severe pain right hip on palpation. No LOC reported."
            ),
            vitals_raw="HR=72, BP=118/76mmHg, sat 94%, T 37.1C, RR 16",
        ),
        "ground_truth": {
            "heart_rate": 72,
            "systolic_bp": 118,
            "diastolic_bp": 76,
            "spo2": 94,
            "temperature": 37.1,
            "respiratory_rate": 16,
        },
    },
]


VITAL_KEYS = ["heart_rate", "systolic_bp", "diastolic_bp", "spo2", "temperature", "respiratory_rate"]
TOLERANCES = {
    "heart_rate": 2,
    "systolic_bp": 2,
    "diastolic_bp": 2,
    "spo2": 1,
    "temperature": 0.2,
    "respiratory_rate": 1,
}


def _parse_agent_json(content: str) -> Dict[str, Any]:
    """Extract JSON from agent response, tolerant of markdown fences."""
    content = content.strip()
    # Strip markdown fences
    content = re.sub(r"```(?:json)?", "", content).replace("```", "").strip()
    # Find first { ... }
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def grade(action: Action, case: Dict[str, Any]) -> Reward:
    """
    Score: F1-style per vital.
    Each vital is worth 1/N of the total score.
    A vital is 'correct' if within tolerance of ground truth.
    Partial credit: 0.5 if present but wrong, 0 if missing/null.
    """
    gt = case["ground_truth"]
    parsed = _parse_agent_json(action.content)

    scores: Dict[str, float] = {}
    breakdown: Dict[str, float] = {}

    for key in VITAL_KEYS:
        gt_val = gt.get(key)
        pred_val = parsed.get(key)

        if gt_val is None:
            # Ground truth doesn't have it; skip
            continue

        if pred_val is None:
            scores[key] = 0.0
            breakdown[key] = 0.0
        else:
            try:
                diff = abs(float(pred_val) - float(gt_val))
                tol = TOLERANCES.get(key, 1)
                if diff <= tol:
                    scores[key] = 1.0
                elif diff <= tol * 3:
                    scores[key] = 0.5
                else:
                    scores[key] = 0.0
            except (TypeError, ValueError):
                scores[key] = 0.0
            breakdown[key] = scores[key]

    total_score = sum(scores.values()) / len(scores) if scores else 0.0
    correct = sum(1 for v in scores.values() if v == 1.0)
    total = len(scores)

    feedback_parts = []
    for k, v in scores.items():
        status = "✓" if v == 1.0 else ("~" if v == 0.5 else "✗")
        feedback_parts.append(f"{status} {k}: pred={parsed.get(k)} gt={gt.get(k)}")

    return Reward(
        value=round(total_score, 4),
        breakdown=breakdown,
        feedback=f"Vitals extracted: {correct}/{total} within tolerance\n" + "\n".join(feedback_parts),
        done=True,
    )


def make_observation(case_idx: int, step: int = 1) -> Tuple[Observation, Dict]:
    case = CASES[case_idx % len(CASES)]
    obs = Observation(
        task_id=TASK_ID,
        task_description=DESCRIPTION,
        patient=case["patient"],
        step=step,
        max_steps=1,
    )
    return obs, case
