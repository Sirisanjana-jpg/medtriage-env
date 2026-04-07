import os
import sys
import json
import time
import traceback
 
# ── Load .env file if present ─────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, fall back to system env vars
 
# ── Config from environment ───────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))
 
if not HF_TOKEN:
    print(
        "ERROR: HF_TOKEN not set.\n"
        "  For Groq:   set HF_TOKEN=gsk_your_key in .env\n"
        "  For OpenAI: set HF_TOKEN=sk-proj-your_key in .env",
        file=sys.stderr
    )
    sys.exit(1)
 
# ── OpenAI client (works with Groq, OpenAI, Together, etc.) ───────
from openai import OpenAI
 
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)
 
# ── Import MedTriageEnv ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.environment import MedTriageEnv, TASK_ORDER, TASKS
from env.models import Action
 
 
# ── System prompt (works well for Llama and GPT models) ───────────
SYSTEM_PROMPT = (
    "You are an expert emergency medicine physician and clinical pharmacist. "
    "You have 20 years of experience in emergency triage and pharmacology. "
    "Analyze the patient case carefully and respond ONLY with a valid JSON object "
    "exactly matching the requested schema. "
    "Do not include any explanation, markdown formatting, or text outside the JSON. "
    "Start your response directly with { and end with }."
)
 
 
def call_llm(prompt: str, max_retries: int = 2) -> str:
    """Call the LLM API and return the response text. Retries on failure."""
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [retry {attempt+1}] API error: {e}. Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise
 
 
def run_task(task_id: str, seed: int = 42) -> dict:
    """Run all cases for one task. Returns per-case scores and mean."""
    mod = TASKS[task_id]
    n_cases = len(mod.CASES)
    case_results = []
 
    for case_idx in range(n_cases):
        env = MedTriageEnv(task_id=task_id, case_idx=case_idx, seed=seed)
        obs = env.reset()
        prompt = obs.to_prompt()
        patient_id = obs.patient.patient_id
 
        t0 = time.time()
        try:
            response = call_llm(prompt)
        except Exception as e:
            response = "{}"
            print(f"  LLM call failed for {task_id} case {case_idx}: {e}", file=sys.stderr)
        elapsed = round(time.time() - t0, 2)
 
        action = Action(task_id=task_id, content=response)
        _, reward, done, info = env.step(action)
 
        step_log = {
            "type":             "STEP",
            "task_id":          task_id,
            "case_idx":         case_idx,
            "patient_id":       patient_id,
            "score":            reward.value,
            "reward_breakdown": reward.breakdown,
            "feedback":         reward.feedback,
            "latency_s":        elapsed,
            "model":            MODEL_NAME,
        }
        print(json.dumps(step_log), flush=True)
 
        bar = "#" * int(reward.value * 20) + "-" * (20 - int(reward.value * 20))
        print(f"  [{bar}] {reward.value:.4f}  {task_id} case {case_idx} ({patient_id}) [{elapsed}s]", file=sys.stderr)
 
        case_results.append(reward.value)
 
    return {
        "task_id":    task_id,
        "scores":     case_results,
        "mean_score": round(sum(case_results) / len(case_results), 4),
        "n_cases":    n_cases,
    }
 
 
def main():
    run_id = f"medtriage_{int(time.time())}"
    seed = 42
 
    print(json.dumps({
        "type":      "START",
        "run_id":    run_id,
        "model":     MODEL_NAME,
        "api_base":  API_BASE_URL,
        "tasks":     TASK_ORDER,
        "seed":      seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }), flush=True)
 
    print(f"\nMedTriageEnv Baseline — {MODEL_NAME}", file=sys.stderr)
    print(f"API: {API_BASE_URL}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
 
    all_results = {}
    total_scores = []
    errors = []
 
    difficulty = {
        "T1_vitals":                 "Easy  ",
        "T2_drug_interactions":      "Medium",
        "T3_differential_diagnosis": "Hard  ",
    }
 
    for task_id in TASK_ORDER:
        print(f"\n[{difficulty[task_id]}] {task_id}", file=sys.stderr)
        try:
            result = run_task(task_id, seed=seed)
            all_results[task_id] = result
            total_scores.extend(result["scores"])
            print(f"  -> mean score: {result['mean_score']:.4f}", file=sys.stderr)
        except Exception as e:
            err = {"task_id": task_id, "error": str(e), "traceback": traceback.format_exc()}
            errors.append(err)
            print(json.dumps({"type": "STEP", "task_id": task_id, "error": str(e)}), flush=True)
            print(f"  ERROR: {e}", file=sys.stderr)
 
    overall_mean = round(sum(total_scores) / len(total_scores), 4) if total_scores else 0.0
 
    print(json.dumps({
        "type":               "END",
        "run_id":             run_id,
        "model":              MODEL_NAME,
        "overall_mean_score": overall_mean,
        "task_results": {
            tid: {"mean_score": r["mean_score"], "scores": r["scores"], "n_cases": r["n_cases"]}
            for tid, r in all_results.items()
        },
        "errors":    errors,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }), flush=True)
 
    print("\n" + "=" * 60, file=sys.stderr)
    print("FINAL SCORES", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for task_id, result in all_results.items():
        diff = difficulty.get(task_id, "?")
        scores_str = ", ".join(f"{s:.4f}" for s in result["scores"])
        print(f"  [{diff}] {task_id}", file=sys.stderr)
        print(f"           scores: [{scores_str}]", file=sys.stderr)
        print(f"           mean:   {result['mean_score']:.4f}", file=sys.stderr)
 
    print(f"\n  OVERALL MEAN: {overall_mean:.4f}", file=sys.stderr)
 
    if errors:
        print(f"\n  ERRORS ({len(errors)}):", file=sys.stderr)
        for e in errors:
            print(f"    - {e['task_id']}: {e['error']}", file=sys.stderr)
 
    print("=" * 60, file=sys.stderr)
    return 0 if not errors else 1
 
 
if __name__ == "__main__":
    sys.exit(main())