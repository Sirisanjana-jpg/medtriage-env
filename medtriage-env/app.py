"""
MedTriageEnv — FastAPI server exposing OpenEnv-compatible REST API.
Fix: reset() accepts empty/null body — all fields optional with defaults.
"""
from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import MedTriageEnv, TASK_ORDER, TASKS
from env.models import Action

app = FastAPI(
    title="MedTriageEnv",
    description="Clinical Decision Support RL Environment — OpenEnv compliant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, MedTriageEnv] = {}


def _get_env(session_id: str = "default") -> MedTriageEnv:
    if session_id not in _sessions:
        env = MedTriageEnv(task_id="T1_vitals", case_idx=0, seed=42)
        env.reset()
        _sessions[session_id] = env
    return _sessions[session_id]


@app.get("/health")
def health():
    return {"status": "ok", "env": "MedTriageEnv", "version": "1.0.0", "timestamp": time.time()}


@app.get("/")
def index():
    html = """<!DOCTYPE html><html><head><title>MedTriageEnv</title>
    <style>body{font-family:system-ui,sans-serif;max-width:800px;margin:60px auto;padding:0 20px}
    code{background:#f5f5f5;padding:2px 6px;border-radius:4px}
    pre{background:#f5f5f5;padding:16px;border-radius:8px;overflow-x:auto}
    .easy{background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:12px;font-size:.8em}
    .medium{background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:12px;font-size:.8em}
    .hard{background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:12px;font-size:.8em}
    </style></head><body>
    <h1>🏥 MedTriageEnv</h1>
    <p><strong>Clinical Decision Support RL Environment</strong> — OpenEnv compliant v1.0.0</p>
    <h2>Tasks</h2><ul>
    <li><code>T1_vitals</code> <span class="easy">Easy</span> — Extract vital signs from nurse notes</li>
    <li><code>T2_drug_interactions</code> <span class="medium">Medium</span> — Flag drug-drug interactions</li>
    <li><code>T3_differential_diagnosis</code> <span class="hard">Hard</span> — Rank differential diagnoses</li>
    </ul>
    <h2>API</h2><p><a href="/docs">Swagger UI</a> | <a href="/openenv.yaml">openenv.yaml</a></p>
    </body></html>"""
    return HTMLResponse(html)


@app.post("/reset")
async def reset(request: Request):
    """Reset environment. Accepts empty body, null body, or JSON body. All fields optional."""
    try:
        raw = await request.body()
        data = json.loads(raw) if raw and len(raw.strip()) > 2 else {}
    except Exception:
        data = {}

    task_id    = str(data.get("task_id",    "T1_vitals"))
    case_idx   = int(data.get("case_idx",   0))
    seed       = int(data.get("seed",       42))
    session_id = str(data.get("session_id", "default"))

    if task_id not in TASKS:
        task_id = "T1_vitals"

    env = MedTriageEnv(task_id=task_id, case_idx=case_idx, seed=seed)
    _sessions[session_id] = env
    obs = env.reset()

    return {
        "observation": obs.model_dump(),
        "prompt":      obs.to_prompt(),
        "session_id":  session_id,
        "task_id":     task_id,
        "case_idx":    case_idx,
    }


@app.post("/step")
async def step(request: Request):
    """Submit action. Accepts empty body gracefully."""
    try:
        raw = await request.body()
        data = json.loads(raw) if raw and len(raw.strip()) > 2 else {}
    except Exception:
        data = {}

    content    = str(data.get("content", data.get("action", "{}"))) or "{}"
    session_id = str(data.get("session_id", "default"))

    env = _get_env(session_id)
    state_before = env.state()

    if state_before.done:
        env.reset()
        state_before = env.state()

    action = Action(task_id=state_before.task_id, content=content)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError:
        env.reset()
        obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


@app.get("/state")
def state(session_id: str = "default"):
    env = _get_env(session_id)
    return env.state().model_dump()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "T1_vitals",               "difficulty": "easy",   "description": "Extract vital signs from unstructured nurse notes",        "num_cases": len(TASKS["T1_vitals"].CASES),               "reward_range": [0.0, 1.0]},
            {"id": "T2_drug_interactions",     "difficulty": "medium", "description": "Identify clinically significant drug-drug interactions",   "num_cases": len(TASKS["T2_drug_interactions"].CASES),     "reward_range": [0.0, 1.0]},
            {"id": "T3_differential_diagnosis","difficulty": "hard",   "description": "Rank differential diagnoses for emergency presentations",  "num_cases": len(TASKS["T3_differential_diagnosis"].CASES),"reward_range": [0.0, 1.0]},
        ]
    }


@app.get("/openenv.yaml", response_class=HTMLResponse)
def serve_yaml():
    try:
        with open("openenv.yaml") as f:
            content = f.read()
        return HTMLResponse(content=content, media_type="text/yaml")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
