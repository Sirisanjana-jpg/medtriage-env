"""
validate.py — Pre-submission validation for MedTriageEnv.
Checks OpenEnv spec compliance, grader ranges, API endpoints, 
Dockerfile presence, and inference.py structure.
Run: python validate.py
"""
import sys
import os
import json
import importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []

def check(name, condition, detail=""):
    icon = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  {icon} {name}" + (f": {detail}" if detail else ""))
    return condition


print("=" * 60)
print("MedTriageEnv — Pre-submission Validation")
print("=" * 60)
print()

# ── 1. File structure ──────────────────────────────────────────
print("1. File structure")
check("env/models.py exists", os.path.exists("env/models.py"))
check("env/environment.py exists", os.path.exists("env/environment.py"))
check("app.py exists", os.path.exists("app.py"))
check("inference.py exists", os.path.exists("inference.py"))
check("openenv.yaml exists", os.path.exists("openenv.yaml"))
check("Dockerfile exists", os.path.exists("Dockerfile"))
check("requirements.txt exists", os.path.exists("requirements.txt"))
check("README.md exists", os.path.exists("README.md"))
check("tasks/task1_vitals.py exists", os.path.exists("tasks/task1_vitals.py"))
check("tasks/task2_drug_interactions.py exists", os.path.exists("tasks/task2_drug_interactions.py"))
check("tasks/task3_differential_dx.py exists", os.path.exists("tasks/task3_differential_dx.py"))
print()

# ── 2. Pydantic models ─────────────────────────────────────────
print("2. Typed Pydantic models")
try:
    from env.models import Observation, Action, Reward, EnvironmentState, PatientRecord
    check("Observation model importable", True)
    check("Action model importable", True)
    check("Reward model importable", True)
    check("EnvironmentState model importable", True)
    # Test instantiation
    r = Reward(value=0.5, breakdown={"a": 0.5}, feedback="ok", done=False)
    check("Reward(value=0.5) instantiates", True)
    try:
        r_bad = Reward(value=1.5)
        check("Reward rejects value > 1.0", False, "Should have raised ValidationError")
    except Exception:
        check("Reward rejects value > 1.0", True)
except Exception as e:
    check("Models import", False, str(e))
print()

# ── 3. Environment API ─────────────────────────────────────────
print("3. Environment API (step/reset/state)")
try:
    from env.environment import MedTriageEnv
    from env.models import Action as A

    for task_id in ["T1_vitals", "T2_drug_interactions", "T3_differential_diagnosis"]:
        env = MedTriageEnv(task_id=task_id, case_idx=0, seed=42)
        obs = env.reset()
        check(f"{task_id}: reset() returns Observation", isinstance(obs, Observation))
        state = env.state()
        check(f"{task_id}: state() returns EnvironmentState", isinstance(state, EnvironmentState))
        check(f"{task_id}: initial state.done == False", state.done == False)
        check(f"{task_id}: initial state.step == 0", state.step == 0)
        action = A(task_id=task_id, content="{}")
        obs2, reward, done, info = env.step(action)
        check(f"{task_id}: step() returns 4-tuple", True)
        check(f"{task_id}: reward in [0.0, 1.0]", 0.0 <= reward.value <= 1.0, f"got {reward.value}")
        check(f"{task_id}: done is bool", isinstance(done, bool))
        check(f"{task_id}: info is dict", isinstance(info, dict))
except Exception as e:
    check("Environment API", False, str(e))
    import traceback; traceback.print_exc()
print()

# ── 4. Task graders ────────────────────────────────────────────
print("4. Task graders — 3 tasks, scores in [0.0, 1.0]")
try:
    from tasks import task1_vitals, task2_drug_interactions, task3_differential_dx
    from env.models import Action as A

    task_mods = {
        "T1_vitals": task1_vitals,
        "T2_drug_interactions": task2_drug_interactions,
        "T3_differential_diagnosis": task3_differential_dx,
    }

    check("3 tasks defined", len(task_mods) >= 3, f"found {len(task_mods)}")

    for task_id, mod in task_mods.items():
        n = len(mod.CASES)
        check(f"{task_id}: has >= 1 case", n >= 1, f"{n} cases")
        for i in range(n):
            obs, case = mod.make_observation(i)
            action = A(task_id=task_id, content="{}")
            reward = mod.grade(action, case)
            check(
                f"{task_id} case {i}: grader score in [0.0,1.0]",
                0.0 <= reward.value <= 1.0,
                f"got {reward.value}"
            )
except Exception as e:
    check("Graders", False, str(e))
    import traceback; traceback.print_exc()
print()

# ── 5. Perfect score test ──────────────────────────────────────
print("5. Perfect-answer grader validation")
try:
    env = MedTriageEnv(task_id="T1_vitals", case_idx=0)
    env.reset()
    perfect = json.dumps({"heart_rate": 112, "systolic_bp": 145, "diastolic_bp": 92, "spo2": 88, "temperature": 38.4, "respiratory_rate": 24})
    _, r, _, _ = env.step(A(task_id="T1_vitals", content=perfect))
    check("T1 perfect answer = 1.0", r.value == 1.0, f"got {r.value}")

    env = MedTriageEnv(task_id="T2_drug_interactions", case_idx=0)
    env.reset()
    perfect2 = json.dumps({"interactions": [
        {"drug_a": "warfarin", "drug_b": "aspirin", "severity": "major", "mechanism": "bleeding", "recommendation": "avoid"},
        {"drug_a": "lisinopril", "drug_b": "potassium", "severity": "moderate", "mechanism": "hyperkalemia", "recommendation": "monitor"},
    ]})
    _, r, _, _ = env.step(A(task_id="T2_drug_interactions", content=perfect2))
    check("T2 perfect answer = 1.0", r.value == 1.0, f"got {r.value}")

    env = MedTriageEnv(task_id="T3_differential_diagnosis", case_idx=0)
    env.reset()
    perfect3 = json.dumps({
        "differentials": [
            {"rank": 1, "diagnosis": "Subarachnoid hemorrhage", "icd10": "I60.9", "reasoning": "thunderclap", "urgency": "emergent"},
            {"rank": 2, "diagnosis": "Meningitis", "icd10": "G03.9", "reasoning": "nuchal rigidity", "urgency": "emergent"},
            {"rank": 3, "diagnosis": "Hypertensive emergency", "icd10": "I10", "reasoning": "BP elevation", "urgency": "urgent"},
        ],
        "recommended_workup": ["CT head", "lumbar puncture", "CT angiography"],
        "disposition": "admit"
    })
    _, r, _, _ = env.step(A(task_id="T3_differential_diagnosis", content=perfect3))
    check("T3 perfect answer = 1.0", r.value == 1.0, f"got {r.value}")
except Exception as e:
    check("Perfect score test", False, str(e))
print()

# ── 6. FastAPI endpoints ───────────────────────────────────────
print("6. FastAPI REST API")
try:
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)

    r = client.get("/health")
    check("GET /health returns 200", r.status_code == 200)

    r = client.get("/tasks")
    check("GET /tasks returns 200", r.status_code == 200)
    check("GET /tasks returns 3 tasks", len(r.json().get("tasks", [])) == 3)

    r = client.post("/reset", json={"task_id": "T1_vitals", "case_idx": 0, "session_id": "val"})
    check("POST /reset returns 200", r.status_code == 200)
    check("POST /reset returns observation", "observation" in r.json())
    check("POST /reset returns prompt", "prompt" in r.json())

    r = client.get("/state?session_id=val")
    check("GET /state returns 200", r.status_code == 200)

    r = client.post("/step", json={"content": "{}", "session_id": "val"})
    check("POST /step returns 200", r.status_code == 200)
    check("POST /step returns reward", "reward" in r.json())
    check("POST /step reward in [0,1]", 0.0 <= r.json()["reward"]["value"] <= 1.0)
except Exception as e:
    check("FastAPI", False, str(e))
print()

# ── 7. openenv.yaml ────────────────────────────────────────────
print("7. openenv.yaml")
try:
    import yaml
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    check("openenv.yaml is valid YAML", True)
    check("has 'name' field", "name" in spec)
    check("has 'version' field", "version" in spec)
    check("has 'tasks' field", "tasks" in spec)
    check("has 3+ tasks in spec", len(spec.get("tasks", [])) >= 3)
except ImportError:
    # yaml not installed, do basic check
    with open("openenv.yaml") as f:
        content = f.read()
    check("openenv.yaml readable", len(content) > 100)
    check("contains 'name:'", "name:" in content)
    check("contains 'tasks:'", "tasks:" in content)
except Exception as e:
    check("openenv.yaml", False, str(e))
print()

# ── 8. inference.py structure ──────────────────────────────────
print("8. inference.py structure")
with open("inference.py") as f:
    inf_src = f.read()
check("inference.py has [START] log", '"START"' in inf_src or "'START'" in inf_src)
check("inference.py has [STEP] log", '"STEP"' in inf_src or "'STEP'" in inf_src)
check("inference.py has [END] log", '"END"' in inf_src or "'END'" in inf_src)
check("inference.py uses API_BASE_URL", "API_BASE_URL" in inf_src)
check("inference.py uses MODEL_NAME", "MODEL_NAME" in inf_src)
check("inference.py uses HF_TOKEN", "HF_TOKEN" in inf_src)
check("inference.py uses OpenAI client", "OpenAI" in inf_src)
print()

# ── 9. Dockerfile ──────────────────────────────────────────────
print("9. Dockerfile")
with open("Dockerfile") as f:
    df = f.read()
check("FROM python", "FROM python" in df)
check("EXPOSE 7860", "7860" in df)
check("HEALTHCHECK present", "HEALTHCHECK" in df)
check("non-root user", "useradd" in df or "USER" in df)
print()

# ── Summary ────────────────────────────────────────────────────
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
failed = [name for name, ok, _ in results if not ok]

print("=" * 60)
print(f"RESULTS: {passed}/{total} checks passed")
if failed:
    print(f"\nFailed checks:")
    for f in failed:
        print(f"  ❌ {f}")
else:
    print("🎉 All checks passed! Ready to submit.")
print("=" * 60)

sys.exit(0 if not failed else 1)
