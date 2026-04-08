# inference.py
# THE file judges run to evaluate your project
# Must be in ROOT folder
# Must complete in under 20 minutes
# Must print a final score between 0.0 and 1.0

import os
import random
from openai import OpenAI
from env.client import LiveFeatureBudgetClient
from env.models import Action

# ─────────────────────────────────────────
# SETUP — Read environment variables
# ─────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
SERVER_URL   = os.getenv("SERVER_URL",   "http://127.0.0.1:8000")

# ─────────────────────────────────────────
# SETUP — OpenAI client for AI decisions
# ─────────────────────────────────────────

openai_client = OpenAI(
    base_url = API_BASE_URL,
    api_key  = HF_TOKEN if HF_TOKEN else "dummy-key"
)

# ─────────────────────────────────────────
# SETUP — Environment client
# ─────────────────────────────────────────

env_client = LiveFeatureBudgetClient(base_url=SERVER_URL)


# ─────────────────────────────────────────
# AGENT — Decides action based on observation
# Uses AI model to make smart decisions
# Falls back to rule-based if model fails
# ─────────────────────────────────────────

def agent_decide(obs) -> int:
    """
    Given current observation
    ask AI model which action to take
    Returns action_id: 0, 1, 2, or 3
    """

    # Build a simple prompt describing the situation
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
You are an AI product manager.
You must decide how to allocate budget across AI features.

Current situation:
{feature_text}
Budget remaining: {obs.budget_remaining}
Step: {obs.step}

Choose ONE action:
0 → Fund Feature 0 (first feature) more
1 → Fund Feature 1 (second feature) more
2 → Fund Feature 2 (third feature) more — only if 3 features exist
3 → Cut budget from worst performing feature

Rules:
- If budget is low, prefer action 3 (cut worst)
- If a feature has high retention but low usage, fund it more
- Always return ONLY a single number: 0, 1, 2, or 3

Your answer (single digit only):
"""

    try:
        response = openai_client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 5
        )
        answer = response.choices[0].message.content.strip()
        action_id = int(answer[0])
        if action_id not in [0, 1, 2, 3]:
            action_id = random.randint(0, 3)

    except Exception:
        # If model fails — use simple rule based decision
        action_id = rule_based_decide(obs)

    return action_id


# ─────────────────────────────────────────
# FALLBACK — Rule based agent
# Used when AI model is not available
# ─────────────────────────────────────────

def rule_based_decide(obs) -> int:
    """
    Simple rules when AI model not available:
    - If budget is below 30% → cut worst (action 3)
    - Else → fund feature with highest retention
    """
    budget_ratio = obs.budget_remaining / 1000.0

    if budget_ratio < 0.30:
        return 3  # Cut worst feature

    # Find feature with highest retention
    best_idx = 0
    best_retention = 0.0
    for i, f in enumerate(obs.features):
        if f.retention_impact > best_retention:
            best_retention = f.retention_impact
            best_idx = i

    return min(best_idx, 3)


# ─────────────────────────────────────────
# RUN ONE TASK
# Runs a full episode for given task
# Returns average reward for that task
# ─────────────────────────────────────────

def run_task(task_name: str) -> float:
    print(f"\n{'='*50}")
    print(f"  RUNNING TASK: {task_name.upper()}")
    print(f"{'='*50}")

    rewards = []

    # Reset with correct task name
    obs = env_client.reset(task=task_name)
    print(f"  Episode started — budget: {obs.budget_remaining}")
    print(f"  Features: {[f.name for f in obs.features]}")

    step = 0
    while not obs.done:
        action_id = agent_decide(obs)
        action    = Action(action_id=action_id)
        obs       = env_client.step(action)
        rewards.append(obs.reward)
        print(f"  Step {step+1}: action={action_id}  reward={obs.reward}  budget={obs.budget_remaining}  done={obs.done}")
        step += 1

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    avg_reward = round(avg_reward, 4)

    print(f"\n  Task {task_name} complete")
    print(f"  Steps taken    : {step}")
    print(f"  Average reward : {avg_reward}")

    return avg_reward


# ─────────────────────────────────────────
# MAIN — Run all 3 tasks and print final score
# ─────────────────────────────────────────

def main():
    print("\n" + "="*50)
    print("  LIVE FEATURE BUDGET ALLOCATOR")
    print("  Starting inference run...")
    print("="*50)

    all_scores = []

    # Run easy task
    easy_score = run_task("easy")
    all_scores.append(easy_score)

    # Run medium task
    medium_score = run_task("medium")
    all_scores.append(medium_score)

    # Run hard task
    hard_score = run_task("hard")
    all_scores.append(hard_score)

    # Final score = average of all 3 tasks
    final_score = round(sum(all_scores) / len(all_scores), 4)

    print("\n" + "="*50)
    print("  FINAL RESULTS")
    print("="*50)
    print(f"  Easy   score : {easy_score}")
    print(f"  Medium score : {medium_score}")
    print(f"  Hard   score : {hard_score}")
    print(f"  FINAL  score : {final_score}")
    print("="*50)

    return final_score


if __name__ == "__main__":
    main()