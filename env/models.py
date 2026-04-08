# models.py
# Defines the language of our environment
# Every other file imports from here

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────
# 1. ONE FEATURE (Chat / Autocomplete / Agent)
# ─────────────────────────────────────────

@dataclass
class FeatureState:
    name: str                  # "chat", "autocomplete", "agent"
    cost_per_request: float    # how expensive per use
    success_rate: float        # 0.0 to 1.0 — how often it works
    retention_impact: float    # 0.0 to 1.0 — do users stay
    usage_volume: int          # how many users using it


# ─────────────────────────────────────────
# 2. WHAT AGENT SEES EVERY STEP
# ─────────────────────────────────────────

@dataclass
class Observation:
    features: list             # list of FeatureState objects
    budget_remaining: float    # how much budget is left today
    step: int                  # which step we are on
    done: bool                 # is the episode over?
    reward: float              # score from last action (0.0 to 1.0)


# ─────────────────────────────────────────
# 3. WHAT AGENT SENDS BACK
# ─────────────────────────────────────────

@dataclass
class Action:
    action_id: int
    # 0 → fund Feature A (Chat) more
    # 1 → fund Feature B (Autocomplete) more
    # 2 → fund Feature C (Agent) more
    # 3 → cut budget from worst feature


# ─────────────────────────────────────────
# 4. EPISODE METADATA
# ─────────────────────────────────────────

@dataclass
class EpisodeState:
    episode_id: str            # unique ID for this episode
    step_count: int            # how many steps taken so far
    task_name: str             # "easy", "medium", or "hard"
    total_reward: float        # cumulative reward so far