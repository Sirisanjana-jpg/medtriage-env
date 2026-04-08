import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set.", file=sys.stderr)
    sys.exit(1)

from openai import OpenAI
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.environment import MedTriageEnv, TASK_ORDER, TASKS
from env.models import Action

SYSTEM_PROMPT = (
    "You are an expert emergency medicine physician and clinical pharmacist. "
    "Analyze the patient case and respond ONLY with a valid JSON object "
    "matching the requested schema. No explanation, no markdown, just JSON."
)

TASK_NAMES = {
    "T1_vitals":                 "vital_signs_extraction",
    "T2_drug_interactions":      "drug_interaction_flagging",
    "T3_differential_diagnosis": "differential_diagnosis_ranking",
}


def clamp(score):
    return max(0.001, min(0.999, round(score, 4)))


def call_llm(prompt):
    for attempt in range(3):
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
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


def main():
    overall_scores = []

    for task_id in TASK_ORDER:
        task_name = TASK_NAMES[task_id]
        mod = TASKS[task_id]
        n_cases = len(mod.CASES)

        print(f"[START] task={task_name} model={MODEL_NAME} cases={n_cases}", flush=True)

        task_scores = []

        for case_idx in range(n_cases):
            step_num = case_idx + 1

            env = MedTriageEnv(task_id=task_id, case_idx=case_idx, seed=42)
            obs = env.reset()
            prompt = obs.to_prompt()

            try:
                response = call_llm(prompt)
            except Exception as e:
                print(f"[STEP] task={task_name} step={step_num} reward=0.001 done=True", flush=True)
                task_scores.append(0.001)
                continue

            action = Action(task_id=task_id, content=response)
            _, reward, done, info = env.step(action)

            score = clamp(reward.value)
            task_scores.append(score)

            print(f"[STEP] task={task_name} step={step_num} reward={score} done={done}", flush=True)

        task_mean = clamp(sum(task_scores) / len(task_scores)) if task_scores else 0.001
        overall_scores.extend(task_scores)

        print(f"[END] task={task_name} score={task_mean} steps={n_cases}", flush=True)

    overall = clamp(sum(overall_scores) / len(overall_scores)) if overall_scores else 0.001
    print(f"\nOVERALL MEAN SCORE: {overall}", file=sys.stderr)


if __name__ == "__main__":
    main()