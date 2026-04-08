# app.py
# The web server
# Wraps environment.py and exposes it over the internet
# Judges ping this. inference.py talks to this.
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env.environment import LiveFeatureBudgetEnv
from env.models import Action
import os

app = FastAPI(
    title       = "Live Feature Budget Allocator",
    description = "RL Environment for AI Product Decision Making",
    version     = "1.0.0"
)

# ─────────────────────────────────────────
# ENVIRONMENT — starts as easy
# Gets recreated on each reset() call
# ─────────────────────────────────────────

TASK = os.getenv("TASK", "easy")
env  = LiveFeatureBudgetEnv(task=TASK)


class ActionInput(BaseModel):
    action_id: int

class ResetInput(BaseModel):
    task: str = "easy"


# ─────────────────────────────────────────
# POST /reset
# Now accepts optional task parameter
# ─────────────────────────────────────────

@app.post("/reset")
def reset(reset_input: ResetInput = None):
    global env

    # If task sent in request body — use it
    # Otherwise use env variable default
    if reset_input and reset_input.task in ["easy", "medium", "hard"]:
        task = reset_input.task
    else:
        task = TASK

    # Recreate environment with correct task
    env = LiveFeatureBudgetEnv(task=task)
    obs = env.reset()

    return {
        "features": [
            {
                "name"            : f.name,
                "cost_per_request": f.cost_per_request,
                "success_rate"    : f.success_rate,
                "retention_impact": f.retention_impact,
                "usage_volume"    : f.usage_volume
            }
            for f in obs.features
        ],
        "budget_remaining": obs.budget_remaining,
        "step"            : obs.step,
        "done"            : obs.done,
        "reward"          : obs.reward
    }


# ─────────────────────────────────────────
# POST /step
# ─────────────────────────────────────────

@app.post("/step")
def step(action_input: ActionInput):
    if action_input.action_id not in [0, 1, 2, 3]:
        raise HTTPException(
            status_code = 400,
            detail      = "action_id must be 0, 1, 2, or 3"
        )

    action = Action(action_id=action_input.action_id)
    obs    = env.step(action)

    return {
        "features": [
            {
                "name"            : f.name,
                "cost_per_request": f.cost_per_request,
                "success_rate"    : f.success_rate,
                "retention_impact": f.retention_impact,
                "usage_volume"    : f.usage_volume
            }
            for f in obs.features
        ],
        "budget_remaining": obs.budget_remaining,
        "step"            : obs.step,
        "done"            : obs.done,
        "reward"          : obs.reward
    }


# ─────────────────────────────────────────
# GET /state
# ─────────────────────────────────────────

@app.get("/state")
def state():
    s = env.state()
    return {
        "episode_id"  : s.episode_id,
        "step_count"  : s.step_count,
        "task_name"   : s.task_name,
        "total_reward": s.total_reward
    }


# ─────────────────────────────────────────
# GET /
# ─────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status" : "live",
        "message": "Live Feature Budget Allocator is running"
    }