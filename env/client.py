# client.py
# The waiter between inference.py and the server
# Handles all HTTP communication
# inference.py never sees raw HTTP — only clean Python objects

import requests
from env.models import FeatureState, Observation, Action, EpisodeState


# ─────────────────────────────────────────
# THE CLIENT CLASS
# ─────────────────────────────────────────

class LiveFeatureBudgetClient:

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        """
        base_url = where the server is running
        Local testing : http://127.0.0.1:8000
        HF Spaces     : https://your-space-url.hf.space
        """
        self.base_url = base_url.rstrip("/")


    # ─────────────────────────────────────
    # HELPER — Parse JSON into Observation
    # ─────────────────────────────────────

    def _parse_observation(self, data: dict) -> Observation:
        """
        Converts raw JSON from server
        into a proper Observation object
        """
        features = []
        for f in data["features"]:
            features.append(FeatureState(
                name             = f["name"],
                cost_per_request = f["cost_per_request"],
                success_rate     = f["success_rate"],
                retention_impact = f["retention_impact"],
                usage_volume     = f["usage_volume"]
            ))

        return Observation(
            features         = features,
            budget_remaining = data["budget_remaining"],
            step             = data["step"],
            done             = data["done"],
            reward           = data["reward"]
        )


    # ─────────────────────────────────────
    # reset()
    # Calls POST /reset on the server
    # Returns Observation object
    # ─────────────────────────────────────

    def reset(self, task: str = "easy") -> Observation:
        payload  = {"task": task}
        response = requests.post(
            f"{self.base_url}/reset",
            json = payload
        )
        response.raise_for_status()
        return self._parse_observation(response.json())
    


    
    # ─────────────────────────────────────
    # step()
    # Sends action to POST /step
    # Returns new Observation object
    # ─────────────────────────────────────

    def step(self, action: Action) -> Observation:
        payload  = {"action_id": action.action_id}
        response = requests.post(
            f"{self.base_url}/step",
            json = payload
        )
        response.raise_for_status()
        return self._parse_observation(response.json())


    # ─────────────────────────────────────
    # state()
    # Calls GET /state on the server
    # Returns EpisodeState object
    # ─────────────────────────────────────

    def state(self) -> EpisodeState:
        response = requests.get(f"{self.base_url}/state")
        response.raise_for_status()
        data = response.json()

        return EpisodeState(
            episode_id   = data["episode_id"],
            step_count   = data["step_count"],
            task_name    = data["task_name"],
            total_reward = data["total_reward"]
        )