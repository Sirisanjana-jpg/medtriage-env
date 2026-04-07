"""
Task 3 (Hard): Differential Diagnosis Ranking
The agent receives a full patient case and must produce a ranked list of
differential diagnoses. Grader uses NDCG (normalized discounted cumulative gain)
— correct diagnoses at rank 1 score higher than at rank 3.
"""
from __future__ import annotations
import re
import json
import math
from typing import Any, Dict, List, Tuple
from env.models import PatientRecord, Observation, Action, Reward


TASK_ID = "T3_differential_diagnosis"
DESCRIPTION = (
    "You are an emergency physician. Based on the complete patient presentation below, "
    "provide a differential diagnosis. Return a JSON object with:\n"
    "{\n"
    '  "differentials": [\n'
    '    {"rank": 1, "diagnosis": "...", "icd10": "...", "reasoning": "...", "urgency": "emergent|urgent|non-urgent"},\n'
    '    {"rank": 2, ...},\n'
    '    {"rank": 3, ...}\n'
    "  ],\n"
    '  "recommended_workup": ["test1", "test2", ...],\n'
    '  "disposition": "admit|discharge|observation"\n'
    "}\n"
    "Rank from most to least likely. Include at least 3 diagnoses."
)

CASES: List[Dict[str, Any]] = [
    {
        "patient": PatientRecord(
            patient_id="P201",
            age=55,
            sex="M",
            chief_complaint="Sudden severe headache",
            nurse_notes=(
                "55yo male presents with sudden onset 'worst headache of my life' that started "
                "30 minutes ago. No prior headache history. HR 88, BP 178/102, SpO2 99%, T 37.0C, RR 16. "
                "Mild nuchal rigidity on exam. No focal neuro deficits. GCS 15. Photophobia present."
            ),
            medications=["Lisinopril 10mg"],
            history="Hypertension. Non-smoker. No recent trauma.",
        ),
        "ground_truth": {
            "primary": ["subarachnoid hemorrhage", "sah", "subarachnoid haemorrhage", "intracranial hemorrhage"],
            "secondary": ["meningitis", "bacterial meningitis", "hypertensive emergency", "migraine", "thunderclap headache"],
            "critical_diagnosis": "subarachnoid hemorrhage",
            "critical_tests": ["ct head", "ct scan", "lumbar puncture", "lp", "ct angiography", "cta"],
            "expected_disposition": "admit",
        },
    },
    {
        "patient": PatientRecord(
            patient_id="P202",
            age=28,
            sex="F",
            chief_complaint="Right lower quadrant pain",
            nurse_notes=(
                "28yo female G1P0 at 8 weeks gestation presents with right lower quadrant pain "
                "and vaginal spotting for 6 hours. Pain is sharp, 8/10. Last normal menstrual period "
                "8 weeks ago. HR 110, BP 92/60, SpO2 98%, T 37.2C. Rebound tenderness present. "
                "urine hCG positive. Shoulder tip pain reported."
            ),
            medications=["Prenatal vitamins"],
            history="Gravida 1. Prior history of PID.",
        ),
        "ground_truth": {
            "primary": ["ectopic pregnancy", "ruptured ectopic", "tubal ectopic"],
            "secondary": ["appendicitis", "ovarian torsion", "threatened miscarriage", "pelvic inflammatory disease"],
            "critical_diagnosis": "ectopic pregnancy",
            "critical_tests": ["pelvic ultrasound", "transvaginal ultrasound", "tvus", "beta hcg", "quantitative hcg", "ultrasound"],
            "expected_disposition": "admit",
        },
    },
    {
        "patient": PatientRecord(
            patient_id="P203",
            age=68,
            sex="M",
            chief_complaint="Acute confusion and fever",
            nurse_notes=(
                "68yo male with DM2 brought by wife for acute confusion over 6 hours. "
                "Fever at home. On exam: T 39.6C, HR 118, BP 88/52, RR 28, SpO2 92% on 4L O2. "
                "Diaphoretic, warm. Oriented only to person. Foley inserted: urine cloudy, foul-smelling. "
                "Glucose 310 mg/dL. Skin intact. No obvious source of trauma."
            ),
            medications=[
                "Metformin 1000mg twice daily",
                "Glipizide 5mg daily",
                "Lisinopril 20mg daily",
                "Atorvastatin 40mg nightly",
            ],
            history="DM2 x 15 years, HTN, CKD stage 2. Lives at home with wife.",
        ),
        "ground_truth": {
            "primary": ["urosepsis", "sepsis", "septic shock", "urinary tract infection", "urosepsis from uti"],
            "secondary": ["pneumonia", "diabetic ketoacidosis", "hyperosmolar hyperglycemic state", "hhs", "dka", "endocarditis"],
            "critical_diagnosis": "sepsis",
            "critical_tests": ["blood cultures", "urine culture", "urinalysis", "lactate", "cbc", "bmp", "chest xray", "ecg"],
            "expected_disposition": "admit",
        },
    },
]


def _parse_differentials(content: str) -> Dict:
    content = re.sub(r"```(?:json)?", "", content).replace("```", "").strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        return {}


def _normalize(s: str) -> str:
    return s.lower().strip()


def _matches_any(text: str, targets: List[str]) -> bool:
    t = _normalize(text)
    return any(_normalize(tgt) in t or t in _normalize(tgt) for tgt in targets)


def _ndcg(ranked_scores: List[float], ideal_scores: List[float]) -> float:
    """
    Normalized Discounted Cumulative Gain.
    ranked_scores: relevance of each predicted item (0 or 1)
    ideal_scores: best possible ordering
    """
    def dcg(scores):
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores))

    actual = dcg(ranked_scores)
    ideal = dcg(sorted(ideal_scores, reverse=True))
    return actual / ideal if ideal > 0 else 0.0


def grade(action: Action, case: Dict[str, Any]) -> Reward:
    """
    Multi-component score:
    1. NDCG score on diagnosis ranking (40%)
    2. Critical diagnosis in top 3 (30%)
    3. Key workup tests mentioned (20%)
    4. Correct disposition (10%)
    """
    gt = case["ground_truth"]
    parsed = _parse_differentials(action.content)

    differentials = parsed.get("differentials", [])
    workup = [w.lower() for w in parsed.get("recommended_workup", [])]
    disposition = _normalize(parsed.get("disposition", ""))

    # --- 1. NDCG on top-3 diagnosis ranking ---
    ranked_scores = []
    for i, diff in enumerate(differentials[:3]):
        diag = _normalize(str(diff.get("diagnosis", "")))
        if _matches_any(diag, gt["primary"]):
            ranked_scores.append(1.0)   # primary = full relevance
        elif _matches_any(diag, gt["secondary"]):
            ranked_scores.append(0.5)   # secondary = partial
        else:
            ranked_scores.append(0.0)

    # Pad to 3 if fewer
    while len(ranked_scores) < 3:
        ranked_scores.append(0.0)

    ideal = sorted(ranked_scores, reverse=True)
    ndcg_score = _ndcg(ranked_scores, ideal) if any(s > 0 for s in ranked_scores) else 0.0

    # --- 2. Critical diagnosis in top 3 ---
    critical_found = False
    critical_rank = None
    for i, diff in enumerate(differentials[:3]):
        diag = _normalize(str(diff.get("diagnosis", "")))
        if _matches_any(diag, gt["primary"]):
            critical_found = True
            critical_rank = i + 1
            break

    critical_score = 0.0
    if critical_found:
        # Rank 1 = 1.0, Rank 2 = 0.75, Rank 3 = 0.5
        critical_score = [1.0, 0.75, 0.5][critical_rank - 1]

    # --- 3. Key workup tests ---
    critical_tests = gt["critical_tests"]
    tests_found = sum(
        1 for t in critical_tests
        if any(_normalize(t) in w or w in _normalize(t) for w in workup)
    )
    test_score = min(tests_found / max(len(critical_tests), 1), 1.0)
    # Require at least 2 critical tests for full credit
    test_score = min(tests_found / 2.0, 1.0)

    # --- 4. Disposition ---
    disposition_score = 1.0 if gt["expected_disposition"] in disposition else 0.0

    # Weighted total
    total = (
        0.40 * ndcg_score +
        0.30 * critical_score +
        0.20 * test_score +
        0.10 * disposition_score
    )

    diag_list = [str(d.get("diagnosis", "?")) for d in differentials[:3]]
    feedback = (
        f"Diagnoses: {diag_list}\n"
        f"NDCG={ndcg_score:.3f} | Critical_found={critical_found} (rank={critical_rank}) "
        f"| Tests={tests_found}/{len(critical_tests)} | Disposition={disposition}(expected={gt['expected_disposition']})"
    )

    return Reward(
        value=round(total, 4),
        breakdown={
            "ndcg": round(ndcg_score, 4),
            "critical_diagnosis": round(critical_score, 4),
            "workup_tests": round(test_score, 4),
            "disposition": disposition_score,
        },
        feedback=feedback,
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
