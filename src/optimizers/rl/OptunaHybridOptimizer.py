import numpy as np
import random
from typing import Dict, List, Any, Tuple
from collections import defaultdict, deque
from optimizers.BaseOptimizer import BaseOptimizer


class CategoricalBandit:
    """Multi-Armed Bandit für kategoriale Parameter (inspired by your GradientBanditOptimizer)"""

    def __init__(self, param_name: str, choices: List, learning_rate: float = 0.3):
        self.param_name = param_name
        self.choices = choices
        self.learning_rate = learning_rate

        # Gradient Bandit: Preferences für jeden Arm
        self.preferences = np.zeros(len(choices))
        self.counts = np.zeros(len(choices))
        self.total_rewards = np.zeros(len(choices))

        # Baseline für relative Rewards
        self.baseline_reward = 0.0
        self.baseline_count = 0

    def softmax_select(self, temperature: float = 1.0) -> int:
        """Softmax selection basierend auf Preferences"""
        try:
            if temperature <= 0:
                temperature = 0.1

            # Prevent overflow by subtracting max
            max_pref = np.max(self.preferences)
            adjusted_prefs = self.preferences - max_pref

            exp_prefs = np.exp(adjusted_prefs / temperature)
            total_exp = np.sum(exp_prefs)

            if total_exp <= 0 or np.isnan(total_exp) or np.isinf(total_exp):
                # Fallback to uniform selection
                return random.randint(0, len(self.choices) - 1)

            probabilities = exp_prefs / total_exp

            # Check for valid probabilities
            if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
                return random.randint(0, len(self.choices) - 1)

            return np.random.choice(len(self.choices), p=probabilities)

        except Exception as e:
            # Fallback to random selection
            return random.randint(0, len(self.choices) - 1)

    def ucb_select(self, total_pulls: int, c: float = 2.0) -> int:
        """UCB selection als Alternative"""
        try:
            if total_pulls <= 0:
                return random.randint(0, len(self.choices) - 1)

            ucb_values = []
            for i in range(len(self.choices)):
                if self.counts[i] <= 0:
                    return i  # Unvisited arms have priority

                mean_reward = self.total_rewards[i] / self.counts[i]

                # Safe log calculation
                log_term = np.log(max(1, total_pulls)) / max(1, self.counts[i])
                confidence = c * np.sqrt(log_term)

                ucb_value = mean_reward + confidence

                # Check for valid UCB value
                if np.isnan(ucb_value) or np.isinf(ucb_value):
                    ucb_value = mean_reward

                ucb_values.append(ucb_value)

            if not ucb_values:
                return random.randint(0, len(self.choices) - 1)

            return np.argmax(ucb_values)

        except Exception as e:
            return random.randint(0, len(self.choices) - 1)

    def update(self, arm_idx: int, reward: float):
        """Update both Gradient Bandit and UCB statistics"""
        try:
            # Handle invalid inputs
            if np.isnan(reward) or np.isinf(reward):
                reward = 0.0

            if arm_idx < 0 or arm_idx >= len(self.choices):
                return  # Invalid arm index

            # Update baseline
            self.baseline_count += 1
            if self.baseline_count > 0:
                alpha = 1.0 / self.baseline_count
                self.baseline_reward += alpha * (reward - self.baseline_reward)

            # Update arm statistics
            self.counts[arm_idx] += 1
            self.total_rewards[arm_idx] += reward

            # Gradient Bandit update
            reward_signal = reward - self.baseline_reward

            # Update preferences for all arms (softmax gradient)
            for i in range(len(self.preferences)):
                if i == arm_idx:
                    # Selected arm: increase preference if above baseline
                    self.preferences[i] += self.learning_rate * reward_signal
                else:
                    # Other arms: decrease preference proportional to their probability
                    current_temp = 1.0  # Use fixed temperature for gradient calculation

                    # Safe softmax calculation
                    max_pref = np.max(self.preferences)
                    adjusted_prefs = self.preferences - max_pref
                    exp_prefs = np.exp(adjusted_prefs / current_temp)
                    total_exp = np.sum(exp_prefs)

                    if total_exp > 0 and not (
                        np.isnan(total_exp) or np.isinf(total_exp)
                    ):
                        prob = exp_prefs[i] / total_exp
                        if not (np.isnan(prob) or np.isinf(prob)):
                            self.preferences[i] -= (
                                self.learning_rate * reward_signal * prob
                            )

        except Exception as e:
            # Silently handle errors to prevent crashes
            pass

    def get_best_choice(self):
        """Get choice with highest preference/average reward"""
        if np.sum(self.counts) == 0:
            return self.choices[0]

        # Use preference-based selection
        best_idx = np.argmax(self.preferences)
        return self.choices[best_idx]


class OptunaHybridOptimizer(BaseOptimizer):
    """
    Hybrid Optimizer: Optuna TPE + RL Bandits

    Strategy:
    - Kontinuierliche Parameter (int, float): Optuna TPE
    - Kategoriale Parameter: Multi-Armed Bandits (Gradient Bandit + UCB)
    - Adaptive strategy switching basierend auf Performance
    - Parameter importance tracking
    """

    def __init__(
        self,
        hyperparameter_space: Dict,
        # TPE Parameters for continuous params
        n_startup_trials: int = 12,
        tpe_gamma: float = 0.25,
        # Bandit Parameters for categorical params
        bandit_learning_rate: float = 0.3,
        bandit_temperature: float = 1.0,
        temperature_decay: float = 0.99,
        min_temperature: float = 0.2,
        # Strategy Management
        strategy_adaptation_window: int = 15,
        exploitation_threshold: int = 25,  # After X rounds, favor exploitation
    ):
        super().__init__("OptunaHybrid", 1000)

        self.hyperparameter_space = hyperparameter_space
        self.n_startup_trials = n_startup_trials
        self.tpe_gamma = tpe_gamma
        self.bandit_learning_rate = bandit_learning_rate
        self.bandit_temperature = bandit_temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.strategy_adaptation_window = strategy_adaptation_window
        self.exploitation_threshold = exploitation_threshold

        # Separate continuous and categorical parameters
        self.continuous_params = {}
        self.categorical_params = {}
        self._categorize_parameters()

        # TPE component for continuous parameters (simplified implementation)
        self.continuous_history = []  # [(params_dict, score), ...]

        # Bandit components for categorical parameters
        self.categorical_bandits = {}
        for param_name, param_config in self.categorical_params.items():
            self.categorical_bandits[param_name] = CategoricalBandit(
                param_name, param_config["values"], bandit_learning_rate
            )

        # Strategy management
        self.round_count = 0
        self.recent_scores = deque(maxlen=strategy_adaptation_window)
        self.strategy_performance = defaultdict(list)
        self.current_strategy = (
            "hybrid"  # "random", "tpe_focus", "bandit_focus", "hybrid"
        )

        # Performance tracking
        self.parameter_importance = {
            param: 1.0 for param in hyperparameter_space.keys()
        }
        self.last_suggested_params = {}
        self.consecutive_no_improvement = 0

        print(
            f"OptunaHybrid: {len(self.continuous_params)} continuous + {len(self.categorical_params)} categorical params"
        )
        print(f"Continuous: {list(self.continuous_params.keys())}")
        print(f"Categorical: {list(self.categorical_params.keys())}")
        print(f"Strategy: TPE + Gradient Bandits with adaptive switching")

    def _categorize_parameters(self):
        """Separate parameters into continuous and categorical"""
        for param_name, param_config in self.hyperparameter_space.items():
            if param_config["type"] == "categorical":
                self.categorical_params[param_name] = param_config
            else:  # int and float are treated as continuous for TPE
                self.continuous_params[param_name] = param_config

    def _tpe_suggest_continuous(self) -> Dict:
        """TPE-inspired suggestion for continuous parameters"""
        try:
            if len(self.continuous_history) < 3:
                # Random sampling for startup
                return self._random_continuous_params()

            # Simple TPE implementation
            continuous_suggestions = {}

            # Get sorted history by score - SAFE VERSION
            history_list = list(self.continuous_history)  # Convert to list
            sorted_history = sorted(history_list, key=lambda x: x[1], reverse=True)

            # Split into good and bad sets (gamma quantile)
            split_point = max(1, int(len(sorted_history) * self.tpe_gamma))
            good_configs = sorted_history[:split_point]  # Safe slicing on list

            for param_name, param_config in self.continuous_params.items():
                try:
                    # Extract values for this parameter
                    good_values = []
                    for config, score in good_configs:
                        if param_name in config:
                            good_values.append(config[param_name])

                    if not good_values:
                        # Fallback to random
                        continuous_suggestions[param_name] = self._random_param_value(
                            param_name, param_config
                        )
                        continue

                    # TPE logic: sample from good distribution
                    if param_config["type"] == "int":
                        # For integers: sample around good values with small perturbation
                        base_value = random.choice(good_values)
                        noise = random.randint(-2, 2)
                        suggested_value = max(
                            param_config["min"],
                            min(param_config["max"], base_value + noise),
                        )
                        continuous_suggestions[param_name] = int(suggested_value)

                    elif param_config["type"] == "float":
                        # For floats: Gaussian around good values
                        base_value = random.choice(good_values)
                        param_range = param_config["max"] - param_config["min"]
                        if param_range <= 0:
                            continuous_suggestions[param_name] = float(base_value)
                        else:
                            std = param_range * 0.1  # 10% of range
                            noise = random.normalvariate(0, std)
                            suggested_value = max(
                                param_config["min"],
                                min(param_config["max"], base_value + noise),
                            )
                            continuous_suggestions[param_name] = float(suggested_value)

                except Exception as e:
                    # Fallback for any parameter-specific errors
                    continuous_suggestions[param_name] = self._random_param_value(
                        param_name, param_config
                    )

            return continuous_suggestions

        except Exception as e:
            print(f"Error in TPE suggestion: {e}")
            return self._random_continuous_params()

    def _bandit_suggest_categorical(self) -> Dict:
        """Multi-Armed Bandit suggestion for categorical parameters"""
        categorical_suggestions = {}

        try:
            total_pulls = max(
                1,
                sum(
                    bandit.baseline_count
                    for bandit in self.categorical_bandits.values()
                ),
            )

            for param_name, bandit in self.categorical_bandits.items():
                try:
                    if self.round_count < self.n_startup_trials:
                        # Random exploration during startup
                        choice_idx = random.randint(0, len(bandit.choices) - 1)
                    elif self.consecutive_no_improvement > 10:
                        # UCB for more exploration when stuck
                        choice_idx = bandit.ucb_select(total_pulls, c=2.5)
                    else:
                        # Softmax selection (exploitation focused)
                        current_temp = max(
                            self.min_temperature,
                            self.bandit_temperature
                            * (self.temperature_decay**self.round_count),
                        )
                        choice_idx = bandit.softmax_select(current_temp)

                    # Ensure valid choice index
                    choice_idx = max(0, min(len(bandit.choices) - 1, choice_idx))
                    categorical_suggestions[param_name] = bandit.choices[choice_idx]

                except Exception as e:
                    # Fallback to random choice for this parameter
                    categorical_suggestions[param_name] = random.choice(bandit.choices)

        except Exception as e:
            # Fallback: generate all categorical parameters randomly
            for param_name, param_config in self.categorical_params.items():
                categorical_suggestions[param_name] = random.choice(
                    param_config["values"]
                )

        return categorical_suggestions

    def _random_continuous_params(self) -> Dict:
        """Generate random continuous parameters"""
        params = {}
        for param_name, param_config in self.continuous_params.items():
            params[param_name] = self._random_param_value(param_name, param_config)
        return params

    def _random_param_value(self, param_name: str, param_config: Dict):
        """Generate random value for a parameter"""
        if param_config["type"] == "int":
            return random.randint(param_config["min"], param_config["max"])
        elif param_config["type"] == "float":
            return random.uniform(param_config["min"], param_config["max"])
        elif param_config["type"] == "categorical":
            return random.choice(param_config["values"])

    def _adapt_strategy(self):
        """Adapt optimization strategy based on recent performance"""
        try:
            if len(self.recent_scores) < 5:
                self.current_strategy = "hybrid"
                return

            # Convert deque to list for safe indexing
            recent_scores_list = list(self.recent_scores)

            # Check recent improvement
            if len(recent_scores_list) >= 5:
                recent_best = max(recent_scores_list[-5:])
                if len(recent_scores_list) > 5:
                    older_best = max(recent_scores_list[:-5])
                else:
                    older_best = 0
                recent_improvement = recent_best - older_best
            else:
                recent_improvement = 0

            # Strategy decision logic
            if self.round_count < self.n_startup_trials:
                self.current_strategy = "random"
            elif self.consecutive_no_improvement > 15:
                # Stuck: more exploration
                self.current_strategy = (
                    "random" if random.random() < 0.3 else "bandit_focus"
                )
            elif (
                recent_improvement < 0.005
                and self.round_count > self.exploitation_threshold
            ):
                # Slow progress: focus on exploitation
                self.current_strategy = (
                    "tpe_focus"
                    if len(self.continuous_params) > len(self.categorical_params)
                    else "bandit_focus"
                )
            else:
                # Normal progression: hybrid approach
                self.current_strategy = "hybrid"

        except Exception as e:
            print(f"Warning: Error in strategy adaptation: {e}")
            self.current_strategy = "hybrid"  # Safe fallback

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Main suggestion method with adaptive strategy"""
        try:
            self.round_count = round_num

            # Adapt strategy based on performance
            self._adapt_strategy()

            # Generate suggestions based on current strategy
            if self.current_strategy == "random":
                # Pure random exploration
                continuous_params = self._random_continuous_params()
                categorical_params = {
                    param_name: random.choice(param_config["values"])
                    for param_name, param_config in self.categorical_params.items()
                }
                decision_info = "RANDOM EXPLORATION"

            elif self.current_strategy == "tpe_focus":
                # Focus on TPE for continuous, random for categorical
                continuous_params = self._tpe_suggest_continuous()
                categorical_params = {
                    param_name: random.choice(param_config["values"])
                    for param_name, param_config in self.categorical_params.items()
                }
                decision_info = "TPE FOCUS"

            elif self.current_strategy == "bandit_focus":
                # Focus on bandits for categorical, random for continuous
                continuous_params = self._random_continuous_params()
                categorical_params = self._bandit_suggest_categorical()
                decision_info = "BANDIT FOCUS"

            else:  # hybrid
                # Full hybrid: TPE + Bandits
                continuous_params = self._tpe_suggest_continuous()
                categorical_params = self._bandit_suggest_categorical()
                decision_info = "HYBRID TPE+BANDITS"

            # Combine all parameters
            suggested_params = {**continuous_params, **categorical_params}

            # Validate that all required parameters are present
            for param_name in self.hyperparameter_space.keys():
                if param_name not in suggested_params:
                    # Add missing parameter with random value
                    param_config = self.hyperparameter_space[param_name]
                    suggested_params[param_name] = self._random_param_value(
                        param_name, param_config
                    )

            # Store for update
            self.last_suggested_params = suggested_params

            # Current temperature for categorical bandits
            current_temp = max(
                self.min_temperature,
                self.bandit_temperature * (self.temperature_decay**self.round_count),
            )

            print(f"Round {round_num}: {decision_info} (temp={current_temp:.3f})")
            print(f"Suggested: {suggested_params}")

            return suggested_params

        except Exception as e:
            print(f"Error in suggest_hyperparameters: {e}")
            # Emergency fallback: pure random
            return {
                param_name: self._random_param_value(param_name, param_config)
                for param_name, param_config in self.hyperparameter_space.items()
            }

    def update(self, hyperparameters: Dict, score: float):
        """Update both TPE and Bandit components"""
        try:
            # Handle NaN scores
            if np.isnan(score):
                score = -1.0
                print(f"Score: NaN (penalized to {score})")

            # Update performance tracking
            self.recent_scores.append(score)
            self.history.append({"params": hyperparameters, "score": score})

            # Safe strategy performance update
            if self.current_strategy in self.strategy_performance:
                self.strategy_performance[self.current_strategy].append(score)
            else:
                self.strategy_performance[self.current_strategy] = [score]

            # Track improvements
            improved = False
            if score > self.best_score:
                old_best = self.best_score
                self.best_score = score
                self.best_params = hyperparameters.copy()
                improved = True
                self.consecutive_no_improvement = 0
                print(f"  *** HYBRID IMPROVEMENT: {score:.4f} (was {old_best:.4f}) ***")
            else:
                self.consecutive_no_improvement += 1
                print(f"  No improvement: {score:.4f} vs best {self.best_score:.4f}")

            # Update TPE history for continuous parameters
            continuous_part = {}
            for k, v in hyperparameters.items():
                if k in self.continuous_params:
                    continuous_part[k] = v

            if continuous_part:
                self.continuous_history.append((continuous_part, score))

            # Update categorical bandits
            for param_name, bandit in self.categorical_bandits.items():
                if param_name in hyperparameters:
                    param_value = hyperparameters[param_name]
                    if param_value in bandit.choices:
                        choice_idx = bandit.choices.index(param_value)
                        bandit.update(choice_idx, score)

            print(f"Score: {score:.4f}, Strategy: {self.current_strategy}")

            # Progress reporting
            if self.round_count % 25 == 0 and self.round_count > self.n_startup_trials:
                self._print_hybrid_progress()

        except Exception as e:
            print(f"Error in OptunaHybrid update: {e}")
            # Continue execution even if update fails

    def _print_hybrid_progress(self):
        """Print detailed progress of both components"""
        try:
            print(f"\n  === Hybrid Progress (Round {self.round_count}) ===")

            # Strategy performance
            print(f"  Strategy performance:")
            for strategy, scores in self.strategy_performance.items():
                if scores:
                    # Convert to list and take safe slice
                    scores_list = list(scores)
                    recent_scores = (
                        scores_list[-10:] if len(scores_list) >= 10 else scores_list
                    )
                    if recent_scores:
                        avg_score = np.mean(recent_scores)
                        print(f"    {strategy}: {avg_score:.4f} (n={len(scores_list)})")

            # Categorical bandit status
            if self.categorical_bandits:
                print(f"  Categorical Bandit Status:")
                for param_name, bandit in self.categorical_bandits.items():
                    if np.sum(bandit.counts) > 0:
                        # Find best choice by preference
                        best_idx = np.argmax(bandit.preferences)
                        if 0 <= best_idx < len(bandit.choices):
                            best_choice = bandit.choices[best_idx]
                            best_pref = bandit.preferences[best_idx]
                            print(
                                f"    {param_name}: best={best_choice} (pref={best_pref:+.2f})"
                            )

            # TPE status
            if len(self.continuous_history) > 5:
                # Safe slicing for continuous history
                recent_history = (
                    self.continuous_history[-10:]
                    if len(self.continuous_history) >= 10
                    else self.continuous_history
                )
                recent_continuous_scores = [score for _, score in recent_history]
                if recent_continuous_scores:
                    print(
                        f"  TPE Continuous: avg_recent={np.mean(recent_continuous_scores):.4f}"
                    )

            print(f"  Consecutive no improvement: {self.consecutive_no_improvement}")
            print()

        except Exception as e:
            print(f"  Error in progress reporting: {e}")
            print()

    def get_component_insights(self) -> Dict:
        """Get detailed insights from both TPE and Bandit components"""
        try:
            insights = {
                "strategy_performance": {},
                "current_strategy": self.current_strategy,
                "consecutive_no_improvement": self.consecutive_no_improvement,
                "categorical_bandits": {},
                "continuous_tpe": {
                    "history_length": len(self.continuous_history),
                    "recent_performance": [],
                },
            }

            # Safe strategy performance extraction
            for strategy, scores in self.strategy_performance.items():
                insights["strategy_performance"][strategy] = list(scores)

            # Categorical bandit insights
            for param_name, bandit in self.categorical_bandits.items():
                try:
                    choice_stats = []
                    for i, choice in enumerate(bandit.choices):
                        if (
                            i < len(bandit.preferences)
                            and i < len(bandit.counts)
                            and i < len(bandit.total_rewards)
                        ):
                            choice_stats.append(
                                {
                                    "choice": choice,
                                    "preference": float(bandit.preferences[i]),
                                    "count": int(bandit.counts[i]),
                                    "avg_reward": float(
                                        bandit.total_rewards[i]
                                        / max(1, bandit.counts[i])
                                    ),
                                }
                            )
                    insights["categorical_bandits"][param_name] = choice_stats
                except Exception as e:
                    insights["categorical_bandits"][param_name] = []

            # TPE insights
            if len(self.continuous_history) >= 5:
                # Safe slicing
                recent_history = (
                    self.continuous_history[-10:]
                    if len(self.continuous_history) >= 10
                    else self.continuous_history
                )
                insights["continuous_tpe"]["recent_performance"] = [
                    float(score) for _, score in recent_history
                ]

            return insights

        except Exception as e:
            # Return minimal safe insights
            return {
                "strategy_performance": {},
                "current_strategy": "hybrid",
                "consecutive_no_improvement": 0,
                "categorical_bandits": {},
                "continuous_tpe": {"history_length": 0, "recent_performance": []},
            }
