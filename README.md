# 🏥 MedTriageEnv

**Clinical Decision Support RL Environment — OpenEnv Compliant v1.0.0**

MedTriageEnv simulates the triage workflow of a real emergency department. Agents must perform tasks that ED nurses and physicians do every shift: extract vital signs from messy nurse notes, flag dangerous drug combinations, and produce ranked differential diagnoses under pressure.

This environment is designed to evaluate language model agents on **real clinical reasoning** — not toy tasks, not simplified games.

---

## Why MedTriage?

Emergency department triage is:
- **High-stakes**: wrong decisions harm patients
- **Information-dense**: nurses write fast, abbreviate, use jargon
- **Knowledge-intensive**: drug interactions and diagnoses require expertise
- **Well-defined**: ground truth exists — vitals are measurable, interactions are documented, diagnoses are validated

This makes it an ideal RL environment: there is real partial credit (3/6 vitals correct = 0.5 reward), deterministic graders, and a natural difficulty progression.

---

## Tasks

### T1: Vital Signs Extraction `[Easy]` 
**Real task**: ED admission triage nurses document vitals in free text.
An agent receives raw nurse notes and must return structured JSON with all vital signs.

**Reward**: F1-style per vital within clinical tolerance
- Heart rate: ±2 bpm
- Blood pressure: ±2 mmHg  
- SpO2: ±1%
- Temperature: ±0.2°C
- Respiratory rate: ±1 breath/min

**Example Input**:
```
Pt arrived via ambulance at 14:32. HR 112, BP 145/92, SpO2 88% on room air, 
temp 38.4C, RR 24. Reports worsening SOB over 3 days.
```

**Expected Output**:
```json
{"heart_rate": 112, "systolic_bp": 145, "diastolic_bp": 92, 
 "spo2": 88, "temperature": 38.4, "respiratory_rate": 24}
```

---

### T2: Drug Interaction Flagging `[Medium]`
**Real task**: Pharmacists and physicians review medication lists for dangerous combinations.
An agent receives a patient's full medication list and must identify all clinically significant interactions.

**Reward**: Precision/Recall F1 on interaction pairs
- Detected correctly: +TP
- Missed: +FN (penalizes recall)  
- Hallucinated: +FP (penalizes precision)

**Interactions covered**: Warfarin+ASA, Statin+Amiodarone, SSRI+Tramadol, Beta-blocker+CCB, ACEi+Potassium, Clopidogrel+PPI, and more.

---

### T3: Differential Diagnosis Ranking `[Hard]`
**Real task**: ER physicians synthesize history, vitals, and exam findings into a differential diagnosis.
An agent receives a complete case and must rank diagnoses by likelihood.

**Reward**: Multi-component NDCG-weighted score:
- **40%** NDCG ranking score (correct dx at rank 1 > rank 3)
- **30%** Critical diagnosis in top 3
- **20%** Key workup tests mentioned
- **10%** Correct disposition (admit/discharge/observation)

**Cases**: Subarachnoid hemorrhage, Ectopic pregnancy, Urosepsis/Septic shock

---

## Observation Space

```
Type: Text (str)
Format: Structured natural language prompt

Fields:
  - Task description and output schema
  - Patient demographics (age, sex)
  - Chief complaint
  - Nurse notes (unstructured text)
  - Medications list
  - Allergies
  - Medical history
  - Step counter
```

## Action Space

```
Type: Text (str)
Format: Natural language response containing a JSON object

T1 Schema: {"heart_rate": int, "systolic_bp": int, ...}
T2 Schema: {"interactions": [{"drug_a": str, "drug_b": str, "severity": str, ...}]}
T3 Schema: {"differentials": [...], "recommended_workup": [...], "disposition": str}
```

## Reward Function

| Task | Metric | Range | Signal Type |
|------|--------|-------|-------------|
| T1 | Per-vital F1 | 0.0–1.0 | Continuous, per vital |
| T2 | Interaction F1 | 0.0–1.0 | Continuous, per pair |
| T3 | Weighted NDCG | 0.0–1.0 | Continuous, rank-aware |

All rewards are **non-binary** — agents receive partial credit for partial success.

---

## API Reference

### REST Endpoints

```bash
# Health check
GET /health

# List tasks
GET /tasks

# Start episode
POST /reset
Body: {"task_id": "T1_vitals", "case_idx": 0, "seed": 42, "session_id": "default"}

# Submit action
POST /step
Body: {"content": "<agent response>", "session_id": "default"}

# Query state
GET /state?session_id=default
```

### Python SDK

```python
from env.environment import MedTriageEnv
from env.models import Action

# Initialize
env = MedTriageEnv(task_id="T1_vitals", case_idx=0, seed=42)

# Reset — get initial observation
obs = env.reset()
print(obs.to_prompt())  # send this to your LLM

# Step — submit agent response
action = Action(task_id="T1_vitals", content='{"heart_rate": 112, ...}')
obs, reward, done, info = env.step(action)

print(f"Score: {reward.value}")        # 0.0–1.0
print(f"Feedback: {reward.feedback}")  # per-vital breakdown
print(f"Done: {done}")

# State
state = env.state()
print(state.episode_reward)
```

---

## Setup

### Option 1: Docker (Recommended)

```bash
git clone https://huggingface.co/spaces/<your-space>/medtriage-env
cd medtriage-env

# Build
docker build -t medtriage-env .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=<your_openai_key> \
  -e MODEL_NAME=gpt-4o-mini \
  -e API_BASE_URL=https://api.openai.com/v1 \
  medtriage-env
```

### Option 2: Local Python

```bash
pip install -r requirements.txt

# Start server
uvicorn app:app --host 0.0.0.0 --port 7860

# Or run inference directly
export HF_TOKEN=<your_openai_api_key>
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=https://api.openai.com/v1
python inference.py
```

---

## Baseline Results

Run with GPT-4o-mini (`gpt-4o-mini`, temperature=0.1, seed=42):

| Task | Difficulty | Score |
|------|-----------|-------|
| T1: Vital Signs Extraction | Easy | ~0.92 |
| T2: Drug Interaction Flagging | Medium | ~0.78 |
| T3: Differential Diagnosis | Hard | ~0.61 |
| **Overall** | | **~0.77** |

### Run Baseline

```bash
export HF_TOKEN=<your_api_key>
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=https://api.openai.com/v1

python inference.py
```

Output format (stdout):
```jsonl
{"type": "START", "run_id": "...", "model": "gpt-4o-mini", ...}
{"type": "STEP", "task_id": "T1_vitals", "case_idx": 0, "score": 0.9167, ...}
{"type": "STEP", "task_id": "T1_vitals", "case_idx": 1, "score": 1.0, ...}
...
{"type": "END", "overall_mean_score": 0.77, "task_results": {...}}
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | ✅ | Your API key (OpenAI or compatible) |
| `MODEL_NAME` | ✅ | Model identifier (e.g. `gpt-4o-mini`) |
| `API_BASE_URL` | ✅ | API endpoint (e.g. `https://api.openai.com/v1`) |

---

## Project Structure

```
medtriage-env/
├── app.py                    # FastAPI server (OpenEnv REST API)
├── inference.py              # Baseline inference script
├── openenv.yaml              # OpenEnv spec file
├── Dockerfile                # HuggingFace Spaces compatible
├── requirements.txt
├── README.md
├── env/
│   ├── __init__.py
│   ├── models.py             # Pydantic: Observation, Action, Reward, State
│   └── environment.py        # MedTriageEnv class (step/reset/state)
└── tasks/
    ├── __init__.py
    ├── task1_vitals.py        # Easy: vital extraction + F1 grader
    ├── task2_drug_interactions.py  # Medium: drug flagging + precision/recall
    └── task3_differential_dx.py   # Hard: differential dx + NDCG grader
```

---

## Deployment (HuggingFace Spaces)

1. Create a new Space: `docker` SDK
2. Push this repository
3. Add Secrets: `HF_TOKEN`, `MODEL_NAME`, `API_BASE_URL`
4. The Space will auto-build and expose the API on port 7860

The Space URL will respond to:
```bash
curl https://<space-url>/health     # {"status": "ok"}
curl -X POST https://<space-url>/reset -d '{"task_id":"T1_vitals"}' 
```

---

## License

MIT License. Clinical case data is synthetically generated for evaluation purposes only. Not for clinical use.
