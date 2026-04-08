# environment.py
# The brain of the project
# All game logic lives here

import random
from env.models import FeatureState, Observation, Action, EpisodeState


# ─────────────────────────────────────────
# HELPER — WORST FEATURE FINDER
# ─────────────────────────────────────────

def find_worst_feature(features: list) -> int:
    """
    Worst feature = lowest (retention_impact - cost_per_request)
    Returns the INDEX of the worst feature
    """
    scores = []
    for f in features:
        score = f.retention_impact - f.cost_per_request
        scores.append(score)
    return scores.index(min(scores))


# ─────────────────────────────────────────
# HELPER — REWARD CALCULATOR
# ─────────────────────────────────────────

def calculate_reward(features: list, budget_remaining: float, starting_budget: float) -> float:
    """
    reward = 0.6 * retention
           + 0.3 * budget_efficiency
           + 0.1 * success_rate
    Clipped between 0.0 and 1.0
    """
    avg_retention = sum(f.retention_impact for f in features) / len(features)
    avg_success   = sum(f.success_rate for f in features) / len(features)
    budget_efficiency = max(0.0, budget_remaining / starting_budget)

    reward = (0.6 * avg_retention) + (0.3 * budget_efficiency) + (0.1 * avg_success)

    # Always clip between 0.0 and 1.0
    return round(max(0.0, min(1.0, reward)), 4)


# ─────────────────────────────────────────
# MAIN ENVIRONMENT CLASS
# ─────────────────────────────────────────

class LiveFeatureBudgetEnv:

    def __init__(self, task: str = "easy"):
        """
        task = "easy" / "medium" / "hard"
        """
        self.task            = task
        self.features        = []
        self.budget          = 0.0
        self.starting_budget = 0.0
        self.step_count      = 0
        self.total_reward    = 0.0
        self.done            = False
        self.episode_id      = str(random.randint(10000, 99999))

        # Max steps per task
        self.max_steps = {
            "easy"  : 5,
            "medium": 8,
            "hard"  : 10
        }[task]


    # ─────────────────────────────────────
    # reset()
    # ─────────────────────────────────────

    def reset(self) -> Observation:
        """
        Start a fresh episode.
        Randomly initialize features based on task difficulty.
        """
        self.step_count   = 0
        self.total_reward = 0.0
        self.done         = False
        self.episode_id   = str(random.randint(10000, 99999))

        # Set budget based on task
        if self.task == "easy":
            self.starting_budget = 1000.0
        elif self.task == "medium":
            self.starting_budget = 800.0
        else:
            self.starting_budget = 1000.0   # hard starts full but shrinks

        self.budget = self.starting_budget

        # Build features based on task
        if self.task == "easy":
            # 2 features only, stable values
            self.features = [
                FeatureState(
                    name             = "chat",
                    cost_per_request = round(random.uniform(0.05, 0.10), 3),
                    success_rate     = round(random.uniform(0.80, 0.95), 3),
                    retention_impact = round(random.uniform(0.75, 0.90), 3),
                    usage_volume     = random.randint(800, 1200)
                ),
                FeatureState(
                    name             = "autocomplete",
                    cost_per_request = round(random.uniform(0.01, 0.03), 3),
                    success_rate     = round(random.uniform(0.65, 0.80), 3),
                    retention_impact = round(random.uniform(0.50, 0.70), 3),
                    usage_volume     = random.randint(400, 800)
                ),
            ]

        elif self.task == "medium":
            # 3 features, slight randomness
            self.features = [
                FeatureState(
                    name             = "chat",
                    cost_per_request = round(random.uniform(0.05, 0.10), 3),
                    success_rate     = round(random.uniform(0.80, 0.95), 3),
                    retention_impact = round(random.uniform(0.75, 0.90), 3),
                    usage_volume     = random.randint(800, 1200)
                ),
                FeatureState(
                    name             = "autocomplete",
                    cost_per_request = round(random.uniform(0.01, 0.03), 3),
                    success_rate     = round(random.uniform(0.65, 0.80), 3),
                    retention_impact = round(random.uniform(0.50, 0.70), 3),
                    usage_volume     = random.randint(400, 800)
                ),
                FeatureState(
                    name             = "agent",
                    cost_per_request = round(random.uniform(0.35, 0.55), 3),
                    success_rate     = round(random.uniform(0.50, 0.70), 3),
                    retention_impact = round(random.uniform(0.80, 0.95), 3),
                    usage_volume     = random.randint(50, 200)
                ),
            ]

        else:
            # hard — 3 features, dynamic
            self.features = [
                FeatureState(
                    name             = "chat",
                    cost_per_request = round(random.uniform(0.05, 0.10), 3),
                    success_rate     = round(random.uniform(0.80, 0.95), 3),
                    retention_impact = round(random.uniform(0.75, 0.90), 3),
                    usage_volume     = random.randint(800, 1200)
                ),
                FeatureState(
                    name             = "autocomplete",
                    cost_per_request = round(random.uniform(0.01, 0.03), 3),
                    success_rate     = round(random.uniform(0.65, 0.80), 3),
                    retention_impact = round(random.uniform(0.50, 0.70), 3),
                    usage_volume     = random.randint(400, 800)
                ),
                FeatureState(
                    name             = "agent",
                    cost_per_request = round(random.uniform(0.35, 0.55), 3),
                    success_rate     = round(random.uniform(0.50, 0.70), 3),
                    retention_impact = round(random.uniform(0.80, 0.95), 3),
                    usage_volume     = random.randint(50, 200)
                ),
            ]

        return Observation(
            features         = self.features,
            budget_remaining = self.budget,
            step             = self.step_count,
            done             = self.done,
            reward           = 0.0
        )


    # ─────────────────────────────────────
    # step()
    # ─────────────────────────────────────

    def step(self, action: Action) -> Observation:
        """
        Agent sends action.
        Environment reacts.
        Returns new observation + reward.
        """

        if self.done:
            # Episode already over
            return self.state()

        action_id = action.action_id

        # ── Apply Action ──────────────────

        if action_id in [0, 1, 2]:
            # Fund a specific feature more
            idx = action_id
            if idx < len(self.features):
                f = self.features[idx]

                # Budget cost = usage * cost per request
                spend = f.usage_volume * f.cost_per_request
                self.budget -= spend

                # Funding improves retention slightly
                f.retention_impact = min(1.0, f.retention_impact + 0.02)
                f.usage_volume     = int(f.usage_volume * 1.05)

        elif action_id == 3:
            # Cut worst feature
            worst_idx = find_worst_feature(self.features)
            f = self.features[worst_idx]

            # Cutting saves budget
            self.budget += f.usage_volume * f.cost_per_request * 0.3

            # But retention drops slightly
            f.retention_impact = max(0.0, f.retention_impact - 0.05)
            f.usage_volume     = int(f.usage_volume * 0.80)

        # ── Hard Task: Budget Shrinks Every Step ──
        if self.task == "hard":
            self.budget -= 50.0

            # Random spike event (20% chance)
            if random.random() < 0.20:
                spike_idx = random.randint(0, len(self.features) - 1)
                self.features[spike_idx].cost_per_request *= 2.0
                self.features[spike_idx].cost_per_request = round(
                    self.features[spike_idx].cost_per_request, 3
                )

        # ── Medium Task: Slight Drift ──
        if self.task in ["medium", "hard"]:
            for f in self.features:
                f.cost_per_request = round(
                    max(0.01, f.cost_per_request + random.uniform(-0.01, 0.01)), 3
                )
                f.success_rate = round(
                    max(0.0, min(1.0, f.success_rate + random.uniform(-0.03, 0.03))), 3
                )

        # ── Update Step ───────────────────
        self.step_count += 1

        # ── Calculate Reward ──────────────
        reward = calculate_reward(self.features, self.budget, self.starting_budget)
        self.total_reward += reward

        # ── Check Done ────────────────────
        if self.budget <= 0:
            self.done = True

        if self.step_count >= self.max_steps:
            self.done = True

        return Observation(
            features         = self.features,
            budget_remaining = round(self.budget, 2),
            step             = self.step_count,
            done             = self.done,
            reward           = reward
        )


    # ─────────────────────────────────────
    # state()
    # ─────────────────────────────────────

    def state(self) -> EpisodeState:
        """
        Returns episode metadata.
        No changes to environment.
        """
        return EpisodeState(
            episode_id   = self.episode_id,
            step_count   = self.step_count,
            task_name    = self.task,
            total_reward = round(self.total_reward, 4)
        )