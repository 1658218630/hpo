import numpy as np
import random
from typing import Dict, List
from optimizers.BaseOptimizer import BaseOptimizer


class ParameterBandit:
    """Ein einzelner Bandit für einen Parameter"""

    def __init__(self, param_name: str, param_config: Dict, region_splits: int = 3):
        self.param_name = param_name
        self.param_config = param_config
        self.region_splits = region_splits

        # Erstelle Regionen/Arms für diesen Parameter
        self.arms = self._create_arms()

        # UCB Statistiken
        self.arm_counts = [0] * len(self.arms)
        self.arm_rewards = [[] for _ in range(len(self.arms))]
        self.total_pulls = 0

    def _create_arms(self) -> List:
        """Erstelle Arms für diesen Parameter"""
        if self.param_config["type"] == "categorical":
            # Jeder kategoriale Wert ist ein Arm
            return self.param_config["values"]

        elif self.param_config["type"] == "int":
            # Teile Integer-Range in Regionen
            min_val, max_val = self.param_config["min"], self.param_config["max"]
            step = max(1, (max_val - min_val) // self.region_splits)
            arms = []
            for i in range(self.region_splits):
                region_min = min_val + i * step
                region_max = min(max_val, min_val + (i + 1) * step)
                arms.append((region_min, region_max))
            return arms

        elif self.param_config["type"] == "float":
            # Teile Float-Range in Regionen
            min_val, max_val = self.param_config["min"], self.param_config["max"]
            step = (max_val - min_val) / self.region_splits
            arms = []
            for i in range(self.region_splits):
                region_min = min_val + i * step
                region_max = min_val + (i + 1) * step
                arms.append((region_min, region_max))
            return arms

    def select_arm_ucb(self, c: float = 2.0) -> int:
        """UCB Arm-Auswahl für diesen Parameter"""
        if self.total_pulls == 0:
            return random.randint(0, len(self.arms) - 1)

        ucb_values = []
        for i in range(len(self.arms)):
            if self.arm_counts[i] == 0:
                return i  # Unbesuchte Arms haben Priorität

            mean_reward = np.mean(self.arm_rewards[i])
            confidence = c * np.sqrt(np.log(self.total_pulls) / self.arm_counts[i])
            ucb_values.append(mean_reward + confidence)

        return np.argmax(ucb_values)

    def sample_from_arm(self, arm_idx: int):
        """Sample konkreten Wert aus einem Arm"""
        arm = self.arms[arm_idx]

        if self.param_config["type"] == "categorical":
            return arm
        elif self.param_config["type"] == "int":
            min_val, max_val = arm
            return random.randint(min_val, max_val)
        elif self.param_config["type"] == "float":
            min_val, max_val = arm
            return random.uniform(min_val, max_val)

    def update(self, arm_idx: int, reward: float):
        """Update Arm mit Reward"""
        self.arm_counts[arm_idx] += 1
        self.arm_rewards[arm_idx].append(reward)
        self.total_pulls += 1


class IndependentBanditOptimizer(BaseOptimizer):
    """
    Independent Parameter Bandits
    Jeder Parameter hat seinen eigenen Multi-Armed Bandit.
    """

    def __init__(
        self,
        hyperparameter_space: Dict,
        region_splits: int = 3,  # Weniger Regionen für besseres Learning
        ucb_c: float = 2.0,  # UCB Exploration Parameter
        epsilon: float = 0.1,  # Kleine epsilon für gelegentliche Random-Exploration
    ):
        super().__init__("IndependentBandit", 100)

        self.hyperparameter_space = hyperparameter_space
        self.region_splits = region_splits
        self.ucb_c = ucb_c
        self.epsilon = epsilon

        # Erstelle einen Bandit pro Parameter
        self.parameter_bandits = {}
        for param_name, param_config in hyperparameter_space.items():
            self.parameter_bandits[param_name] = ParameterBandit(
                param_name, param_config, region_splits
            )

        # Tracking
        self.last_arm_selections = {}
        self.round_count = 0

        # Debug Info
        total_arms = sum(len(bandit.arms) for bandit in self.parameter_bandits.values())
        print(
            f"IndependentBandit: {len(self.parameter_bandits)} parameters, {total_arms} total arms"
        )
        for param_name, bandit in self.parameter_bandits.items():
            print(f"  {param_name}: {len(bandit.arms)} arms")

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Schlage Parameter vor - jeder Parameter wählt unabhängig"""
        self.round_count = round_num

        suggested_params = {}
        selected_arms = {}

        # Jeder Parameter wählt unabhängig seinen besten Arm
        for param_name, bandit in self.parameter_bandits.items():

            if random.random() < self.epsilon:
                # Epsilon-Exploration: zufälliger Arm
                arm_idx = random.randint(0, len(bandit.arms) - 1)
                decision = "RANDOM"
            else:
                # UCB-Auswahl
                arm_idx = bandit.select_arm_ucb(self.ucb_c)
                decision = "UCB"

            # Sample konkreten Wert aus Arm
            value = bandit.sample_from_arm(arm_idx)
            suggested_params[param_name] = value
            selected_arms[param_name] = arm_idx

            # Debug für kategoriale Parameter
            if bandit.param_config["type"] == "categorical":
                print(f"  {param_name}: {decision} -> arm {arm_idx} = {value}")

        # Speichere für Update
        self.last_arm_selections = selected_arms

        print(f"Round {round_num}: Independent Bandit Selection")
        print(f"Suggested: {suggested_params}")

        return suggested_params

    def update(self, hyperparameters: Dict, score: float):
        """Update alle Parameter-Bandits mit dem Score"""

        # Global tracking
        self.history.append({"params": hyperparameters, "score": score})
        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters.copy()
            print(f"  *** NEW BEST: {score:.4f} ***")

        # Update jeden Parameter-Bandit
        for param_name, arm_idx in self.last_arm_selections.items():
            bandit = self.parameter_bandits[param_name]
            bandit.update(arm_idx, score)

        print(f"Score: {score:.4f}")

        # Debug: Zeige Arm-Statistiken für kategoriale Parameter
        if self.round_count % 50 == 0:  # Alle 50 Runden
            self._print_bandit_stats()

    def _print_bandit_stats(self):
        """Zeige Bandit-Statistiken"""
        print(f"\n  === Bandit Stats (Round {self.round_count}) ===")

        for param_name, bandit in self.parameter_bandits.items():
            if bandit.param_config["type"] == "categorical":
                print(f"  {param_name}:")

                # Sortiere Arms nach Performance
                arm_stats = []
                for i, arm_value in enumerate(bandit.arms):
                    if bandit.arm_counts[i] > 0:
                        avg_reward = np.mean(bandit.arm_rewards[i])
                        arm_stats.append((arm_value, avg_reward, bandit.arm_counts[i]))

                arm_stats.sort(key=lambda x: x[1], reverse=True)  # Sort by avg reward

                for arm_value, avg_reward, count in arm_stats[:3]:  # Top 3
                    print(f"    {arm_value}: {avg_reward:.4f} (n={count})")
        print()

    def get_best_parameter_values(self) -> Dict:
        """Hole die aktuell besten Werte für jeden Parameter"""
        best_params = {}

        for param_name, bandit in self.parameter_bandits.items():
            # Finde Arm mit bester Performance
            best_arm_idx = -1
            best_avg_reward = -float("inf")

            for i in range(len(bandit.arms)):
                if bandit.arm_counts[i] > 0:
                    avg_reward = np.mean(bandit.arm_rewards[i])
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        best_arm_idx = i

            if best_arm_idx >= 0:
                # Sample aus bestem Arm
                best_params[param_name] = bandit.sample_from_arm(best_arm_idx)
            else:
                # Fallback: Default-Wert
                if bandit.param_config["type"] == "categorical":
                    best_params[param_name] = bandit.arms[0]
                elif bandit.param_config["type"] == "int":
                    best_params[param_name] = (
                        bandit.param_config["min"] + bandit.param_config["max"]
                    ) // 2
                else:
                    best_params[param_name] = (
                        bandit.param_config["min"] + bandit.param_config["max"]
                    ) / 2

        return best_params
