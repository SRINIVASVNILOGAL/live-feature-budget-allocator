# inference.py
# Judges run this file to evaluate your project
# Must follow strict [START] [STEP] [END] log format

import os
import json
import random
from openai import OpenAI
from env.client import LiveFeatureBudgetClient
from env.models import Action

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     "dummy-key")
SERVER_URL = os.getenv("SERVER_URL", "https://aimlforwolrd-live-feature-budget-allocator.hf.space")

openai_client = OpenAI(
    base_url = API_BASE_URL,
    api_key  = HF_TOKEN if HF_TOKEN else "dummy-key"
)

env_client = LiveFeatureBudgetClient(base_url=SERVER_URL)


# ─────────────────────────────────────────
# STRUCTURED LOGGING — REQUIRED BY JUDGES
# ─────────────────────────────────────────

def log_start(task: str, model: str):
    print(json.dumps({
        "type"  : "START",
        "task"  : task,
        "model" : model
    }), flush=True)


def log_step(step: int, action: int, reward: float, done: bool):
    print(json.dumps({
        "type"  : "STEP",
        "step"  : step,
        "action": action,
        "reward": reward,
        "done"  : done
    }), flush=True)


def log_end(task: str, score: float, steps: int):
    print(json.dumps({
        "type" : "END",
        "task" : task,
        "score": score,
        "steps": steps
    }), flush=True)


# ─────────────────────────────────────────
# AGENT DECISION
# ─────────────────────────────────────────

def agent_decide(obs) -> int:
    feature_text = ""
    for i, f in enumerate(obs.features):
        feature_text += f"""
Feature {i} ({f.name}):
  cost_per_request : {f.cost_per_request}
  success_rate     : {f.success_rate}
  retention_impact : {f.retention_impact}
  usage_volume     : {f.usage_volume}
"""

    prompt = f"""
You are an AI product manager allocating budget across AI features.

Current situation:
{feature_text}
Budget remaining: {obs.budget_remaining}
Step: {obs.step}

Choose ONE action:
0 → Fund Feature 0 (Chat) more
1 → Fund Feature 1 (Autocomplete) more
2 → Fund Feature 2 (Agent) more
3 → Cut worst performing feature

Rules:
- If budget is low prefer action 3
- Fund features with high retention
- Return ONLY a single digit: 0, 1, 2, or 3

Your answer:
"""

    try:
        response = openai_client.chat.completions.create(
            model      = MODEL_NAME,
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 5
        )
        answer    = response.choices[0].message.content.strip()
        action_id = int(answer[0])
        if action_id not in [0, 1, 2, 3]:
            action_id = rule_based_decide(obs)
    except Exception:
        action_id = rule_based_decide(obs)

    return action_id


def rule_based_decide(obs) -> int:
    budget_ratio = obs.budget_remaining / 1000.0
    if budget_ratio < 0.30:
        return 3
    best_idx       = 0
    best_retention = 0.0
    for i, f in enumerate(obs.features):
        if f.retention_impact > best_retention:
            best_retention = f.retention_impact
            best_idx       = i
    return min(best_idx, 3)


# ─────────────────────────────────────────
# RUN ONE TASK
# ─────────────────────────────────────────

def run_task(task_name: str) -> float:
    log_start(task=task_name, model=MODEL_NAME)

    rewards = []
    obs     = env_client.reset(task=task_name)

    step = 0
    while not obs.done:
        action_id = agent_decide(obs)
        action    = Action(action_id=action_id)
        obs       = env_client.step(action)

        rewards.append(obs.reward)
        log_step(
            step   = step + 1,
            action = action_id,
            reward = obs.reward,
            done   = obs.done
        )
        step += 1

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    avg_reward = round(max(0.0, min(1.0, avg_reward)), 4)

    log_end(task=task_name, score=avg_reward, steps=step)

    return avg_reward


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    all_scores = []

    easy_score   = run_task("easy")
    all_scores.append(easy_score)

    medium_score = run_task("medium")
    all_scores.append(medium_score)

    hard_score   = run_task("hard")
    all_scores.append(hard_score)

    final_score = round(sum(all_scores) / len(all_scores), 4)

    print(json.dumps({
        "type"        : "FINAL",
        "easy_score"  : easy_score,
        "medium_score": medium_score,
        "hard_score"  : hard_score,
        "final_score" : final_score
    }), flush=True)

    return final_score


if __name__ == "__main__":
    main()