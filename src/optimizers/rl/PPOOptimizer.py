import numpy as np
import random
from typing import Dict, List
from optimizers.BaseOptimizer import BaseOptimizer


class SimplePolicyNetwork:
    """Einfaches Policy Network für PPO"""

    def __init__(self, input_size: int, output_size: int, learning_rate: float = 0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Kleine Netzwerk-Architektur
        hidden_size = 32

        # Policy Network (gibt Mittelwerte aus)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        # Value Network (schätzt State-Value)
        self.W1_v = np.random.randn(input_size, hidden_size) * 0.1
        self.b1_v = np.zeros((1, hidden_size))
        self.W2_v = np.random.randn(hidden_size, 1) * 0.1
        self.b2_v = np.zeros((1, 1))

    def forward_policy(self, state):
        """Forward pass für Policy (gibt Parameter-Mittelwerte aus)"""
        if state.ndim == 1:
            state = state.reshape(1, -1)

        # Hidden layer
        h1 = np.maximum(0, np.dot(state, self.W1) + self.b1)  # ReLU
        # Output layer (Sigmoid für normalisierte Parameter)
        output = 1 / (1 + np.exp(-(np.dot(h1, self.W2) + self.b2)))  # Sigmoid
        return output

    def forward_value(self, state):
        """Forward pass für Value Network"""
        if state.ndim == 1:
            state = state.reshape(1, -1)

        h1 = np.maximum(0, np.dot(state, self.W1_v) + self.b1_v)
        value = np.dot(h1, self.W2_v) + self.b2_v
        return value[0, 0]

    def update_policy(self, states, actions, advantages, old_probs):
        """PPO Policy Update (vereinfacht)"""
        for i in range(len(states)):
            state = states[i].reshape(1, -1)
            action = actions[i].reshape(1, -1)
            advantage = advantages[i]
            old_prob = old_probs[i]

            # Forward pass
            pred_action = self.forward_policy(state)

            # Berechne neue Wahrscheinlichkeit (vereinfacht als MSE-ähnlich)
            new_prob = np.exp(-np.mean((pred_action - action) ** 2))

            # PPO Clipping (vereinfacht)
            ratio = new_prob / (old_prob + 1e-8)
            clipped_ratio = np.clip(ratio, 0.8, 1.2)  # PPO clip

            # Policy Loss (vereinfacht)
            loss_factor = min(ratio * advantage, clipped_ratio * advantage)

            if loss_factor > 0:  # Nur updaten wenn Vorteil
                # Gradient descent auf Policy
                error = pred_action - action

                # Backward pass (vereinfacht)
                grad_W2 = (
                    np.dot(np.maximum(0, np.dot(state, self.W1) + self.b1).T, error)
                    * loss_factor
                    * self.learning_rate
                )
                grad_b2 = (
                    np.sum(error, axis=0, keepdims=True)
                    * loss_factor
                    * self.learning_rate
                )

                self.W2 -= grad_W2
                self.b2 -= grad_b2

    def update_value(self, states, returns):
        """Value Network Update"""
        for i in range(len(states)):
            state = states[i].reshape(1, -1)
            target = returns[i]

            # Forward pass
            pred_value = self.forward_value(state)

            # MSE Loss
            error = pred_value - target

            # Backward pass für Value Network
            h1 = np.maximum(0, np.dot(state.reshape(1, -1), self.W1_v) + self.b1_v)

            grad_W2_v = h1.T * error * self.learning_rate
            grad_b2_v = error * self.learning_rate
            grad_h1 = np.dot(error, self.W2_v.T)
            grad_W1_v = np.dot(state.T, grad_h1 * (h1 > 0)) * self.learning_rate
            grad_b1_v = (
                np.sum(grad_h1 * (h1 > 0), axis=0, keepdims=True) * self.learning_rate
            )

            self.W2_v -= grad_W2_v
            self.b2_v -= grad_b2_v
            self.W1_v -= grad_W1_v
            self.b1_v -= grad_b1_v


class PPOOptimizer(BaseOptimizer):
    """
    Proximal Policy Optimization für Hyperparameter Optimization
    """

    def __init__(
        self,
        hyperparameter_space: Dict,
        learning_rate: float = 0.003,
        initial_noise_std: float = 0.5,  # Startens mit viel Exploration
        noise_decay: float = 0.995,  # Noise wird langsam reduziert
        min_noise_std: float = 0.1,  # Minimum Noise für Exploration
    ):
        super().__init__("PPO", 1000)  # Größerer Memory für Online-Learning

        self.hyperparameter_space = hyperparameter_space
        self.learning_rate = learning_rate
        self.initial_noise_std = initial_noise_std
        self.current_noise_std = initial_noise_std
        self.noise_decay = noise_decay
        self.min_noise_std = min_noise_std

        # Parameter Info für Normalisierung
        self.param_names = sorted(hyperparameter_space.keys())
        self.param_mins = []
        self.param_maxs = []
        self.param_types = []

        for param_name in self.param_names:
            config = hyperparameter_space[param_name]
            self.param_types.append(config["type"])

            if config["type"] == "categorical":
                self.param_mins.append(0)
                self.param_maxs.append(len(config["values"]) - 1)
            else:
                self.param_mins.append(config["min"])
                self.param_maxs.append(config["max"])

        # State size: normalized_params + score_info + progress
        state_size = (
            len(self.param_names) + 3
        )  # +3 für [current_score, trend, progress]
        action_size = len(self.param_names)  # Direkte Parameter

        # PPO Network
        self.policy_net = SimplePolicyNetwork(state_size, action_size, learning_rate)

        # Online Learning - speichere nur letzte Erfahrung
        self.last_state = None
        self.last_action = None
        self.last_value = None
        self.last_prob = None

        # Tracking
        self.round_count = 0
        self.recent_scores = []
        self.updates_count = 0

        print(
            f"PPO: Online Learning, state_size={state_size}, action_size={action_size}"
        )
        print(f"Noise: {initial_noise_std} -> {min_noise_std} (decay={noise_decay})")
        print(f"Parameters: {self.param_names}")

    def _normalize_params(self, params: Dict) -> np.ndarray:
        """Normalisiere Parameter zu [0,1]"""
        normalized = []
        for i, param_name in enumerate(self.param_names):
            value = params[param_name]

            if self.param_types[i] == "categorical":
                # Categorical: finde Index
                options = self.hyperparameter_space[param_name]["values"]
                idx = options.index(value)
                normalized_val = idx / (len(options) - 1) if len(options) > 1 else 0.5
            else:
                # Numeric: normalisiere zu [0,1]
                min_val, max_val = self.param_mins[i], self.param_maxs[i]
                normalized_val = (value - min_val) / (max_val - min_val)

            normalized.append(np.clip(normalized_val, 0, 1))

        return np.array(normalized)

    def _denormalize_params(self, normalized: np.ndarray) -> Dict:
        """Konvertiere normalisierte Werte zurück zu Parametern"""
        params = {}

        for i, param_name in enumerate(self.param_names):
            norm_val = np.clip(float(normalized[i]), 0, 1)  # ← Convert to native float

            if self.param_types[i] == "categorical":
                # Categorical: wähle nächsten Index
                options = self.hyperparameter_space[param_name]["values"]
                idx = int(round(norm_val * (len(options) - 1)))
                params[param_name] = options[idx]
            elif self.param_types[i] == "int":
                # Integer
                min_val, max_val = self.param_mins[i], self.param_maxs[i]
                value = min_val + norm_val * (max_val - min_val)
                params[param_name] = int(round(value))  # ← Native int
            else:
                # Float
                min_val, max_val = self.param_mins[i], self.param_maxs[i]
                value = min_val + norm_val * (max_val - min_val)
                params[param_name] = float(value)  # ← Convert to native float

        return params

    def _get_current_state(self) -> np.ndarray:
        """Erstelle aktuellen State"""
        # Default Parameter wenn noch keine da sind
        if not hasattr(self, "current_params"):
            # Verwende Default-Werte oder Mitte des Raums
            default_params = {}
            for param_name in self.param_names:
                config = self.hyperparameter_space[param_name]
                if config["type"] == "categorical":
                    default_params[param_name] = config["values"][0]
                elif config["type"] == "int":
                    default_params[param_name] = (config["min"] + config["max"]) // 2
                else:
                    default_params[param_name] = (config["min"] + config["max"]) / 2
            self.current_params = default_params

        # Normalisierte Parameter
        norm_params = self._normalize_params(self.current_params)

        # Score Info
        current_score = self.best_score if self.best_score > -float("inf") else 0.0

        # Trend (letzten 3 Scores)
        if len(self.recent_scores) >= 3:
            trend = (self.recent_scores[-1] - self.recent_scores[-3]) / 2
        else:
            trend = 0.0

        # Progress (0 bis 1)
        progress = min(self.round_count / 100, 1.0)

        # Kombiniere alles
        state = np.concatenate([norm_params, [current_score, trend, progress]])

        return state.astype(np.float32)

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Schlage neue Parameter mit PPO vor"""
        self.round_count = round_num

        # Adaptive Noise (startet hoch, wird reduziert)
        self.current_noise_std = max(
            self.min_noise_std, self.current_noise_std * self.noise_decay
        )

        # Aktuelle State
        state = self._get_current_state()

        if round_num < 3:
            # Erste Runden: Random Exploration
            suggested_params = {}
            for param_name in self.param_names:
                config = self.hyperparameter_space[param_name]
                if config["type"] == "categorical":
                    suggested_params[param_name] = random.choice(config["values"])
                elif config["type"] == "int":
                    suggested_params[param_name] = random.randint(
                        config["min"], config["max"]
                    )
                else:
                    suggested_params[param_name] = random.uniform(
                        config["min"], config["max"]
                    )

            print(f"Round {round_num}: RANDOM EXPLORATION")

            # Speichere für nächstes Update (aber nicht verwenden)
            self.last_state = state

        else:
            # PPO Policy
            policy_mean = self.policy_net.forward_policy(state)[0]

            # Adaptive Exploration Noise
            noise = np.random.normal(0, self.current_noise_std, len(policy_mean))
            action = np.clip(policy_mean + noise, 0, 1)

            # Konvertiere zu tatsächlichen Parametern
            suggested_params = self._denormalize_params(action)

            # Speichere für nächstes Update
            action_prob = np.exp(
                -np.mean(noise**2) / (2 * self.current_noise_std**2)
            )  # Gaussian Probability
            value = self.policy_net.forward_value(state)

            self.last_state = state
            self.last_action = action
            self.last_prob = action_prob
            self.last_value = value

            print(
                f"Round {round_num}: PPO POLICY (noise_std={self.current_noise_std:.3f})"
            )

        # Update current params
        self.current_params = suggested_params

        print(f"Suggested: {suggested_params}")
        return suggested_params

    def update(self, hyperparameters: Dict, score: float):
        """Online PPO Update - lernt sofort nach jeder Evaluation"""
        # Tracking
        self.recent_scores.append(score)
        self.history.append({"params": hyperparameters, "score": score})

        # Update best
        improved = False
        if not np.isnan(score) and score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters.copy()
            improved = True
            print(f"  *** NEW BEST: {score:.4f} ***")

        # Berechne Reward
        if len(self.recent_scores) >= 2:
            # Belohne Verbesserung vs letzter Score
            reward = score - self.recent_scores[-2]

            # Extra Belohnung für neue Bestleistung
            if improved:
                reward += 0.1

        else:
            reward = score  # Erster Score

        # Handle NaN scores
        if np.isnan(score):
            reward = -1.0  # Large penalty for NaN
            print(f"Score: NaN (penalized), Reward: {reward:.4f}")
        else:
            print(f"Score: {score:.4f}, Reward: {reward:.4f}")

        # Online PPO Update (wenn wir vorherige Erfahrung haben)
        if (
            self.last_state is not None
            and self.last_action is not None
            and self.round_count > 3
            and not np.isnan(reward)
        ):  # ← Nur updaten wenn reward nicht NaN

            self._online_ppo_update(reward)
            self.updates_count += 1

            if self.updates_count % 10 == 0:
                print(f"  PPO Updates: {self.updates_count}")

    def _online_ppo_update(self, reward: float):
        """Führe sofortigen PPO Update mit letzter Erfahrung durch"""

        # Aktuelle State (nach Aktion)
        current_state = self._get_current_state()
        current_value = self.policy_net.forward_value(current_state)

        # TD-Target und Advantage
        gamma = 0.95
        td_target = reward + gamma * current_value
        advantage = td_target - self.last_value

        # Normalisierung (einfach)
        if abs(advantage) > 1e-8:
            advantage = advantage / (abs(advantage) + 1e-8)  # Einfache Normalisierung

        # Policy Update (vereinfacht für Online Learning)
        if advantage > 0:  # Nur updaten wenn Vorteil
            # Simuliere minimalen "Batch" für Update
            states = np.array([self.last_state])
            actions = np.array([self.last_action])
            advantages = np.array([advantage])
            old_probs = np.array([self.last_prob])

            self.policy_net.update_policy(states, actions, advantages, old_probs)

        # Value Update
        states = np.array([self.last_state])
        targets = np.array([td_target])
        self.policy_net.update_value(states, targets)
