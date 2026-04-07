"""
MedTriageEnv — FastAPI server exposing OpenEnv-compatible REST API.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /health
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

# In-memory session store (per-session env instances)
_sessions: Dict[str, MedTriageEnv] = {}

# Default env for single-session use
_default_env: Optional[MedTriageEnv] = None


class ResetRequest(BaseModel):
    task_id: str = "T1_vitals"
    case_idx: int = 0
    seed: Optional[int] = 42
    session_id: Optional[str] = "default"


class StepRequest(BaseModel):
    content: str
    session_id: Optional[str] = "default"


def _get_env(session_id: str = "default") -> MedTriageEnv:
    if session_id not in _sessions:
        raise HTTPException(status_code=400, detail=f"No session '{session_id}'. Call /reset first.")
    return _sessions[session_id]


@app.get("/health")
def health():
    return {"status": "ok", "env": "MedTriageEnv", "version": "1.0.0", "timestamp": time.time()}


@app.get("/")
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>MedTriageEnv</title>
    <style>
      body { font-family: system-ui, sans-serif; max-width: 800px; margin: 60px auto; padding: 0 20px; }
      h1 { color: #1a1a1a; } h2 { color: #333; border-bottom: 1px solid #eee; padding-bottom: 8px; }
      code { background: #f5f5f5; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
      pre { background: #f5f5f5; padding: 16px; border-radius: 8px; overflow-x: auto; }
      .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; margin-left: 8px; }
      .easy { background: #d1fae5; color: #065f46; }
      .medium { background: #fef3c7; color: #92400e; }
      .hard { background: #fee2e2; color: #991b1b; }
    </style>
    </head>
    <body>
    <h1>🏥 MedTriageEnv</h1>
    <p><strong>Clinical Decision Support RL Environment</strong> — OpenEnv compliant v1.0.0</p>

    <h2>Tasks</h2>
    <ul>
      <li><code>T1_vitals</code> <span class="badge easy">Easy</span> — Extract vital signs from nurse notes</li>
      <li><code>T2_drug_interactions</code> <span class="badge medium">Medium</span> — Flag drug-drug interactions</li>
      <li><code>T3_differential_diagnosis</code> <span class="badge hard">Hard</span> — Rank differential diagnoses</li>
    </ul>

    <h2>Quick Start</h2>
    <pre>
# 1. Reset
curl -X POST /reset -H "Content-Type: application/json" \\
  -d '{"task_id": "T1_vitals", "case_idx": 0}'

# 2. Step
curl -X POST /step -H "Content-Type: application/json" \\
  -d '{"content": "{\\"heart_rate\\": 112, \\"systolic_bp\\": 145, ...}"}'

# 3. State
curl /state
    </pre>

    <h2>API Docs</h2>
    <p>→ <a href="/docs">Interactive Swagger UI</a></p>
    <p>→ <a href="/openenv.yaml">openenv.yaml spec</a></p>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.post("/reset")
def reset(req: ResetRequest):
    env = MedTriageEnv(task_id=req.task_id, case_idx=req.case_idx, seed=req.seed)
    _sessions[req.session_id] = env
    obs = env.reset()
    return {
        "observation": obs.model_dump(),
        "prompt": obs.to_prompt(),
        "session_id": req.session_id,
    }


@app.post("/step")
def step(req: StepRequest):
    env = _get_env(req.session_id)
    state_before = env.state()
    action = Action(task_id=state_before.task_id, content=req.content)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str = "default"):
    env = _get_env(session_id)
    return env.state().model_dump()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "T1_vitals",
                "difficulty": "easy",
                "description": "Extract vital signs from unstructured nurse notes",
                "num_cases": len(TASKS["T1_vitals"].CASES),
                "reward_range": [0.0, 1.0],
            },
            {
                "id": "T2_drug_interactions",
                "difficulty": "medium",
                "description": "Identify clinically significant drug-drug interactions",
                "num_cases": len(TASKS["T2_drug_interactions"].CASES),
                "reward_range": [0.0, 1.0],
            },
            {
                "id": "T3_differential_diagnosis",
                "difficulty": "hard",
                "description": "Rank differential diagnoses for emergency presentations",
                "num_cases": len(TASKS["T3_differential_diagnosis"].CASES),
                "reward_range": [0.0, 1.0],
            },
        ]
    }


@app.get("/openenv.yaml", response_class=HTMLResponse)
def serve_yaml():
    try:
        with open("openenv.yaml", "r") as f:
            content = f.read()
        return HTMLResponse(content=content, media_type="text/yaml")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
