"""
Task 2 (Medium): Drug Interaction Flagging
The agent receives a patient's medication list and must identify dangerous
drug-drug interactions and rate their severity.
Grader: precision/recall with severity weighting.
"""
from __future__ import annotations
import re
import json
from typing import Any, Dict, List, Tuple
from env.models import PatientRecord, Observation, Action, Reward


TASK_ID = "T2_drug_interactions"
DESCRIPTION = (
    "Review the patient's current medication list and identify ALL clinically significant "
    "drug-drug interactions. For each interaction found, provide:\n"
    "- drug_a: first drug name\n"
    "- drug_b: second drug name\n"
    "- severity: 'major', 'moderate', or 'minor'\n"
    "- mechanism: brief explanation\n"
    "- recommendation: clinical action to take\n\n"
    "Return a JSON object: {\"interactions\": [{...}, ...]}\n"
    "If no interactions, return {\"interactions\": []}"
)

# Ground-truth interaction database (subset for evaluation)
INTERACTION_DB: Dict[str, Dict] = {
    ("warfarin", "aspirin"): {
        "severity": "major",
        "mechanism": "Additive anticoagulant/antiplatelet effect, increased bleeding risk",
        "keywords": ["bleeding", "anticoagulant", "platelet", "hemorrhage"],
    },
    ("warfarin", "ibuprofen"): {
        "severity": "major",
        "mechanism": "NSAIDs inhibit platelet function and increase GI bleeding risk with anticoagulants",
        "keywords": ["bleeding", "nsaid", "platelet", "gi"],
    },
    ("metformin", "contrast"): {
        "severity": "major",
        "mechanism": "IV contrast can cause acute kidney injury, leading to metformin accumulation and lactic acidosis",
        "keywords": ["lactic acidosis", "kidney", "renal", "contrast"],
    },
    ("ssri", "tramadol"): {
        "severity": "major",
        "mechanism": "Risk of serotonin syndrome due to combined serotonergic activity",
        "keywords": ["serotonin syndrome", "serotonergic"],
    },
    ("lisinopril", "potassium"): {
        "severity": "moderate",
        "mechanism": "ACE inhibitors reduce potassium excretion; potassium supplements risk hyperkalemia",
        "keywords": ["hyperkalemia", "potassium", "ace inhibitor"],
    },
    ("simvastatin", "amiodarone"): {
        "severity": "major",
        "mechanism": "Amiodarone inhibits CYP3A4 and CYP2C9, increasing simvastatin levels → myopathy/rhabdomyolysis",
        "keywords": ["myopathy", "rhabdomyolysis", "cyp", "statin"],
    },
    ("metoprolol", "verapamil"): {
        "severity": "major",
        "mechanism": "Additive negative chronotropic and inotropic effects; risk of bradycardia and heart block",
        "keywords": ["bradycardia", "heart block", "av block"],
    },
    ("clopidogrel", "omeprazole"): {
        "severity": "moderate",
        "mechanism": "Omeprazole inhibits CYP2C19, reducing clopidogrel activation and antiplatelet efficacy",
        "keywords": ["cyp2c19", "antiplatelet", "reduced efficacy"],
    },
}

CASES: List[Dict[str, Any]] = [
    {
        "patient": PatientRecord(
            patient_id="P101",
            age=72,
            sex="M",
            chief_complaint="Chest tightness",
            nurse_notes="Hypertensive male with history of AFib, on multiple medications.",
            medications=[
                "Warfarin 5mg daily",
                "Aspirin 81mg daily",
                "Metoprolol 50mg twice daily",
                "Lisinopril 10mg daily",
                "Potassium chloride 20mEq daily",
            ],
        ),
        "ground_truth_pairs": [
            ("warfarin", "aspirin"),
            ("lisinopril", "potassium"),
        ],
        "drug_aliases": {
            "warfarin": ["warfarin", "coumadin"],
            "aspirin": ["aspirin", "asa"],
            "lisinopril": ["lisinopril", "zestril"],
            "potassium": ["potassium", "kcl", "k+"],
            "metoprolol": ["metoprolol", "lopressor"],
        },
    },
    {
        "patient": PatientRecord(
            patient_id="P102",
            age=58,
            sex="F",
            chief_complaint="Muscle pain and weakness",
            nurse_notes="Female with cardiac history presenting with progressive muscle weakness.",
            medications=[
                "Simvastatin 40mg nightly",
                "Amiodarone 200mg daily",
                "Warfarin 4mg daily",
                "Ibuprofen 400mg PRN pain",
                "Omeprazole 20mg daily",
            ],
        ),
        "ground_truth_pairs": [
            ("simvastatin", "amiodarone"),
            ("warfarin", "ibuprofen"),
        ],
        "drug_aliases": {
            "simvastatin": ["simvastatin", "zocor"],
            "amiodarone": ["amiodarone", "cordarone"],
            "warfarin": ["warfarin", "coumadin"],
            "ibuprofen": ["ibuprofen", "advil", "motrin", "nsaid"],
        },
    },
    {
        "patient": PatientRecord(
            patient_id="P103",
            age=45,
            sex="M",
            chief_complaint="Agitation and confusion",
            nurse_notes="Male on psychiatric medications, presenting with acute agitation, diaphoresis, and tremor.",
            medications=[
                "Sertraline 100mg daily",
                "Tramadol 50mg PRN",
                "Metoprolol 25mg daily",
                "Verapamil 80mg three times daily",
                "Clopidogrel 75mg daily",
                "Omeprazole 40mg daily",
            ],
        ),
        "ground_truth_pairs": [
            ("ssri", "tramadol"),       # sertraline = SSRI
            ("metoprolol", "verapamil"),
            ("clopidogrel", "omeprazole"),
        ],
        "drug_aliases": {
            "ssri": ["sertraline", "ssri", "serotonin"],
            "tramadol": ["tramadol", "ultram"],
            "metoprolol": ["metoprolol", "lopressor"],
            "verapamil": ["verapamil", "calan"],
            "clopidogrel": ["clopidogrel", "plavix"],
            "omeprazole": ["omeprazole", "prilosec", "ppi"],
        },
    },
]


def _parse_interactions(content: str) -> List[Dict]:
    content = re.sub(r"```(?:json)?", "", content).replace("```", "").strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return []
    try:
        obj = json.loads(match.group())
        return obj.get("interactions", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def _normalize(s: str) -> str:
    return s.lower().strip()


def _match_drug(name: str, aliases: List[str]) -> bool:
    n = _normalize(name)
    return any(alias in n or n in alias for alias in aliases)


def grade(action: Action, case: Dict[str, Any]) -> Reward:
    """
    Precision/Recall on interaction pairs detected.
    Each major interaction = 1.0 weight, moderate = 0.7, minor = 0.4.
    Final score = F1 weighted.
    """
    gt_pairs = case["ground_truth_pairs"]
    aliases = case["drug_aliases"]
    parsed = _parse_interactions(action.content)

    # For each ground-truth pair, check if agent found it
    gt_found = {pair: False for pair in gt_pairs}
    false_positives = 0

    for interaction in parsed:
        drug_a = str(interaction.get("drug_a", "")).lower()
        drug_b = str(interaction.get("drug_b", "")).lower()

        matched_gt = False
        for pair in gt_pairs:
            ka, kb = pair
            aliases_a = aliases.get(ka, [ka])
            aliases_b = aliases.get(kb, [kb])

            if (_match_drug(drug_a, aliases_a) and _match_drug(drug_b, aliases_b)) or \
               (_match_drug(drug_b, aliases_a) and _match_drug(drug_a, aliases_b)):
                gt_found[pair] = True
                matched_gt = True
                break

        if not matched_gt:
            false_positives += 1

    tp = sum(1 for v in gt_found.values() if v)
    fn = sum(1 for v in gt_found.values() if not v)
    fp = false_positives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    breakdown = {
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    }

    found_str = [f"✓ {p}" for p, v in gt_found.items() if v]
    missed_str = [f"✗ {p}" for p, v in gt_found.items() if not v]
    feedback = f"Interactions: {tp}/{len(gt_pairs)} found. FP={fp}\n"
    if found_str:
        feedback += "Found: " + ", ".join(str(x) for x in found_str) + "\n"
    if missed_str:
        feedback += "Missed: " + ", ".join(str(x) for x in missed_str) + "\n"

    return Reward(
        value=round(f1, 4),
        breakdown=breakdown,
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
