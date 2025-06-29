import numpy as np
import random
from typing import Dict, List
from optimizers.BaseOptimizer import BaseOptimizer


class GradientBanditOptimizer(BaseOptimizer):
    """
    Gradient Bandit mit starkem Exploitation-Fokus
    """

    def __init__(
        self,
        hyperparameter_space: Dict,
        learning_rate: float = 0.5,  # Aggressives Lernen
        initial_exploration: int = 15,  # Nur 15 Random-Runden
        temperature_decay: float = 0.98,  # Softmax wird fokussierter
        min_temperature: float = 0.1,  # Sehr fokussiert
        stuck_threshold: int = 20,  # Reset wenn 20 Runden keine Verbesserung
    ):
        super().__init__("GradientBandit", 1000)

        self.hyperparameter_space = hyperparameter_space
        self.learning_rate = learning_rate
        self.initial_exploration = initial_exploration
        self.temperature = 1.0
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.stuck_threshold = stuck_threshold

        # Parameter Arms - weniger Regions für stärkere Signale
        self.parameter_arms = self._create_parameter_arms()

        # Gradient Bandit: Preferences für jeden Arm
        self.arm_preferences = {}
        self.arm_counts = {}
        self.arm_total_rewards = {}

        # Initialisiere Preferences
        for param_name, arms in self.parameter_arms.items():
            self.arm_preferences[param_name] = np.zeros(len(arms))
            self.arm_counts[param_name] = np.zeros(len(arms))
            self.arm_total_rewards[param_name] = np.zeros(len(arms))

        # Tracking für kontinuierliche Verbesserung
        self.round_count = 0
        self.last_selections = {}
        self.recent_scores = []
        self.baseline_score = 0.0  # Für Gradient Bandit
        self.rounds_since_improvement = 0
        self.improvement_history = []  # Track improvements

        print(f"GradientBandit: Exploitation-focused, temp_decay={temperature_decay}")
        print(
            f"Parameter arms: {[(p, len(a)) for p, a in self.parameter_arms.items()]}"
        )

    def _create_parameter_arms(self) -> Dict[str, List]:
        """Erstelle weniger Arms für stärkere Signale"""
        arms = {}

        for param_name, param_config in self.hyperparameter_space.items():
            if param_config["type"] == "categorical":
                # Jeder kategoriale Wert ist ein Arm
                arms[param_name] = param_config["values"]

            elif param_config["type"] == "int":
                # Nur 2 Regionen für int Parameter
                min_val, max_val = param_config["min"], param_config["max"]
                mid = (min_val + max_val) // 2
                arms[param_name] = [(min_val, mid), (mid + 1, max_val)]

            elif param_config["type"] == "float":
                # Nur 2 Regionen für float Parameter
                min_val, max_val = param_config["min"], param_config["max"]
                mid = (min_val + max_val) / 2
                arms[param_name] = [(min_val, mid), (mid, max_val)]

        return arms

    def _softmax_selection(self, param_name: str) -> int:
        """Softmax Arm Selection basierend auf Preferences"""
        preferences = self.arm_preferences[param_name]

        # Softmax mit Temperature
        exp_prefs = np.exp(preferences / self.temperature)
        probabilities = exp_prefs / np.sum(exp_prefs)

        # Sample basierend auf Wahrscheinlichkeiten
        return np.random.choice(len(probabilities), p=probabilities)

    def _sample_from_arm(self, param_name: str, arm_idx: int):
        """Sample konkreten Wert aus Arm"""
        param_config = self.hyperparameter_space[param_name]
        arm = self.parameter_arms[param_name][arm_idx]

        if param_config["type"] == "categorical":
            return arm
        elif param_config["type"] == "int":
            min_val, max_val = arm
            return random.randint(min_val, max_val)
        elif param_config["type"] == "float":
            min_val, max_val = arm
            return random.uniform(min_val, max_val)

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Suggest mit starkem Exploitation-Fokus"""
        self.round_count = round_num

        # Adaptive Temperature (wird fokussierter über Zeit)
        if round_num > self.initial_exploration:
            self.temperature = max(
                self.min_temperature, self.temperature * self.temperature_decay
            )

        suggested_params = {}
        selected_arms = {}

        if round_num < self.initial_exploration:
            # Minimale Random Exploration
            for param_name, arms in self.parameter_arms.items():
                arm_idx = random.randint(0, len(arms) - 1)
                value = self._sample_from_arm(param_name, arm_idx)
                suggested_params[param_name] = value
                selected_arms[param_name] = arm_idx

            print(
                f"Round {round_num}: INITIAL EXPLORATION ({self.initial_exploration - round_num} left)"
            )

        else:
            # Gradient Bandit Selection
            decision_rationale = []

            for param_name, arms in self.parameter_arms.items():
                # Check for stuck situation
                if self.rounds_since_improvement > self.stuck_threshold:
                    # Force exploration auf schlecht performenden Parameter
                    worst_arm = np.argmin(self.arm_preferences[param_name])
                    arm_idx = worst_arm
                    decision_rationale.append(f"{param_name}:UNSTUCK")
                else:
                    # Normal Softmax Selection
                    arm_idx = self._softmax_selection(param_name)

                    # Show preference strengths
                    prefs = self.arm_preferences[param_name]
                    best_pref = np.max(prefs)
                    selected_pref = prefs[arm_idx]
                    decision_rationale.append(
                        f"{param_name}:SOFTMAX({selected_pref:.2f}/{best_pref:.2f})"
                    )

                value = self._sample_from_arm(param_name, arm_idx)
                suggested_params[param_name] = value
                selected_arms[param_name] = arm_idx

            print(f"Round {round_num}: EXPLOITATION (temp={self.temperature:.3f})")
            print(f"  Decisions: {', '.join(decision_rationale)}")

            # Reset stuck counter wenn wir force exploration gemacht haben
            if self.rounds_since_improvement > self.stuck_threshold:
                self.rounds_since_improvement = 0

        # Store für Update
        self.last_selections = selected_arms

        print(f"Suggested: {suggested_params}")
        return suggested_params

    def update(self, hyperparameters: Dict, score: float):
        """Gradient Bandit Update mit Exploitation-Fokus"""

        # Global tracking
        self.recent_scores.append(score)
        self.history.append({"params": hyperparameters, "score": score})

        # Track improvements für kontinuierliche Kurve
        improved = False
        if score > self.best_score:
            improvement = score - self.best_score
            self.improvement_history.append((self.round_count, improvement))
            self.best_score = score
            self.best_params = hyperparameters.copy()
            improved = True
            self.rounds_since_improvement = 0
            print(f"  *** IMPROVEMENT: +{improvement:.4f} -> {score:.4f} ***")
        else:
            self.rounds_since_improvement += 1

        # Update Baseline (average of recent scores)
        if len(self.recent_scores) >= 5:
            self.baseline_score = np.mean(self.recent_scores[-10:])  # Last 10 scores
        else:
            self.baseline_score = np.mean(self.recent_scores)

        # Gradient Bandit Updates (nur wenn nicht initial exploration)
        if self.round_count >= self.initial_exploration and self.last_selections:
            reward_signal = score - self.baseline_score  # Reward relative to baseline

            for param_name, selected_arm in self.last_selections.items():
                # Update Arm Statistics
                self.arm_counts[param_name][selected_arm] += 1
                self.arm_total_rewards[param_name][selected_arm] += score

                # Gradient Bandit Update
                preferences = self.arm_preferences[param_name]

                # Update preferences für alle Arms
                for arm_idx in range(len(preferences)):
                    if arm_idx == selected_arm:
                        # Selected arm: increase preference if above baseline
                        preferences[arm_idx] += self.learning_rate * reward_signal
                    else:
                        # Other arms: decrease preference (softmax gradient)
                        prob = np.exp(preferences[arm_idx] / self.temperature)
                        prob /= np.sum(np.exp(preferences / self.temperature))
                        preferences[arm_idx] -= (
                            self.learning_rate * reward_signal * prob
                        )

        print(
            f"Score: {score:.4f}, Baseline: {self.baseline_score:.4f}, "
            + f"Signal: {score - self.baseline_score:+.4f}"
        )

        # Show learning progress every 25 rounds
        if self.round_count % 25 == 0 and self.round_count > self.initial_exploration:
            self._print_learning_progress()

    def _print_learning_progress(self):
        """Zeige Lern-Fortschritt"""
        print(f"\n  === Learning Progress (Round {self.round_count}) ===")

        for param_name, preferences in self.arm_preferences.items():
            if len(preferences) <= 5:  # Nur für überschaubare Parameter
                print(f"  {param_name}:")

                # Sortiere Arms nach Preference
                arm_data = []
                for i, (pref, arm) in enumerate(
                    zip(preferences, self.parameter_arms[param_name])
                ):
                    count = self.arm_counts[param_name][i]
                    if count > 0:
                        avg_reward = self.arm_total_rewards[param_name][i] / count
                        arm_data.append((arm, pref, avg_reward, count))

                arm_data.sort(key=lambda x: x[1], reverse=True)  # Sort by preference

                for arm, pref, avg_reward, count in arm_data[:3]:  # Top 3
                    print(
                        f"    {arm}: pref={pref:+.2f}, avg={avg_reward:.4f} (n={count})"
                    )

        # Show improvement trend
        if len(self.improvement_history) >= 3:
            recent_improvements = [imp for _, imp in self.improvement_history[-5:]]
            print(
                f"  Recent improvements: {[f'+{imp:.4f}' for imp in recent_improvements]}"
            )

        print()
