import numpy as np
import random
from typing import Dict, List, Tuple
from optimizers.BaseOptimizer import BaseOptimizer
from scipy.stats import beta, norm
import warnings

warnings.filterwarnings("ignore")


class BayesianParameterModel:
    """Bayesian Model für einen Parameter mit Beta/Gaussian Posterior"""

    def __init__(self, param_name: str, param_config: Dict, num_arms: int = 3):
        self.param_name = param_name
        self.param_config = param_config
        self.num_arms = num_arms

        # Erstelle Arms
        self.arms = self._create_arms()

        if param_config["type"] == "categorical":
            # Beta-Verteilungen für kategoriale Parameter
            self.alpha = np.ones(len(self.arms))  # Beta prior alpha
            self.beta_param = np.ones(len(self.arms))  # Beta prior beta
        else:
            # Gaussian-Verteilungen für numerische Parameter
            self.means = np.zeros(len(self.arms))  # Posterior means
            self.variances = np.ones(len(self.arms))  # Posterior variances
            self.counts = np.zeros(len(self.arms))  # Observation counts
            self.sum_rewards = np.zeros(len(self.arms))  # Sum of rewards

    def _create_arms(self) -> List:
        """Erstelle Arms für Parameter"""
        if self.param_config["type"] == "categorical":
            return self.param_config["values"]

        elif self.param_config["type"] == "int":
            min_val, max_val = self.param_config["min"], self.param_config["max"]
            step = max(1, (max_val - min_val) // self.num_arms)
            arms = []
            for i in range(self.num_arms):
                arm_min = min_val + i * step
                arm_max = min(max_val, min_val + (i + 1) * step)
                arms.append((arm_min, arm_max))
            return arms

        elif self.param_config["type"] == "float":
            min_val, max_val = self.param_config["min"], self.param_config["max"]
            step = (max_val - min_val) / self.num_arms
            arms = []
            for i in range(self.num_arms):
                arm_min = min_val + i * step
                arm_max = min_val + (i + 1) * step
                arms.append((arm_min, arm_max))
            return arms

    def thompson_sampling(self) -> int:
        """Thompson Sampling für Arm-Auswahl"""
        try:
            if self.param_config["type"] == "categorical":
                # Beta-Distribution Thompson Sampling
                samples = []
                for i in range(len(self.arms)):
                    sample = np.random.beta(self.alpha[i], self.beta_param[i])
                    samples.append(sample)
                return int(np.argmax(samples))  # Ensure integer return
            else:
                # Gaussian Thompson Sampling
                samples = []
                for i in range(len(self.arms)):
                    if self.counts[i] > 0:
                        # Posterior Gaussian sampling
                        posterior_mean = float(self.means[i])
                        posterior_var = float(self.variances[i]) / max(
                            1, float(self.counts[i])
                        )
                        sample = np.random.normal(
                            posterior_mean, np.sqrt(posterior_var)
                        )
                    else:
                        # Prior sampling (wide uncertainty)
                        sample = np.random.normal(0, 1)
                    samples.append(sample)
                return int(np.argmax(samples))  # Ensure integer return
        except Exception as e:
            # Fallback to random selection
            return random.randint(0, len(self.arms) - 1)

    def update_posterior(self, arm_idx: int, reward: float):
        """Bayesian Posterior Update"""
        try:
            # Validate arm_idx
            if arm_idx < 0 or arm_idx >= len(self.arms):
                return

            if self.param_config["type"] == "categorical":
                # Beta-Binomial Update
                # Convert reward to success/failure (reward > threshold)
                threshold = 0.55  # Anpassbar
                if reward > threshold:
                    self.alpha[arm_idx] += 1  # Success
                else:
                    self.beta_param[arm_idx] += 1  # Failure
            else:
                # Gaussian Update (Online Bayesian Linear Regression style)
                self.counts[arm_idx] += 1
                self.sum_rewards[arm_idx] += reward

                # Update posterior mean (simple average)
                self.means[arm_idx] = self.sum_rewards[arm_idx] / self.counts[arm_idx]

                # Update posterior variance (decreases with more observations)
                self.variances[arm_idx] = 1.0 / (1.0 + self.counts[arm_idx] * 0.1)
        except Exception as e:
            # Silent error handling
            pass

    def sample_from_arm(self, arm_idx: int):
        """Sample konkreten Wert aus Arm"""
        try:
            # Validate arm_idx
            if arm_idx < 0 or arm_idx >= len(self.arms):
                arm_idx = 0

            arm = self.arms[arm_idx]

            if self.param_config["type"] == "categorical":
                return arm
            elif self.param_config["type"] == "int":
                min_val, max_val = arm
                return random.randint(min_val, max_val)
            elif self.param_config["type"] == "float":
                min_val, max_val = arm
                return random.uniform(min_val, max_val)
        except Exception as e:
            # Fallback: return default value
            if self.param_config["type"] == "categorical":
                return self.param_config["values"][0]
            elif self.param_config["type"] == "int":
                return (self.param_config["min"] + self.param_config["max"]) // 2
            else:
                return (self.param_config["min"] + self.param_config["max"]) / 2

    def get_posterior_stats(self) -> Dict:
        """Hole Posterior-Statistiken für Debugging"""
        try:
            if self.param_config["type"] == "categorical":
                stats = {}
                for i, arm in enumerate(self.arms):
                    alpha_val = float(self.alpha[i])
                    beta_val = float(self.beta_param[i])
                    mean = alpha_val / (alpha_val + beta_val)
                    stats[str(arm)] = {
                        "posterior_mean": float(mean),
                        "alpha": float(alpha_val),
                        "beta": float(beta_val),
                        "confidence": float(alpha_val + beta_val),  # Total observations
                    }
                return stats
            else:
                stats = {}
                for i, arm in enumerate(self.arms):
                    stats[str(arm)] = {
                        "posterior_mean": float(self.means[i]),
                        "posterior_var": float(self.variances[i]),
                        "observations": float(self.counts[i]),
                    }
                return stats
        except Exception as e:
            return {}


class BayesianRLOptimizer(BaseOptimizer):
    """
    Bayesian Reinforcement Learning für HPO

    Verwendet Bayesian Posterior über Parameter-Performance:
    - Thompson Sampling für Exploration vs Exploitation
    - Beta-Verteilungen für kategoriale Parameter
    - Gaussian Posteriors für numerische Parameter
    - Uncertainty-guided Exploration
    """

    def __init__(
        self,
        hyperparameter_space: Dict,
        num_arms_per_param: int = 3,  # Weniger Arms für klarere Signale
        initial_exploration: int = 15,  # Initial Random-Phase
        uncertainty_bonus: float = 0.1,  # Bonus für unsichere Parameter
    ):
        super().__init__("BayesianRL", 1000)

        self.hyperparameter_space = hyperparameter_space
        self.num_arms_per_param = num_arms_per_param
        self.initial_exploration = initial_exploration
        self.uncertainty_bonus = uncertainty_bonus

        # Erstelle Bayesian Models für jeden Parameter
        self.param_models = {}
        for param_name, param_config in hyperparameter_space.items():
            self.param_models[param_name] = BayesianParameterModel(
                param_name, param_config, num_arms_per_param
            )

        # Tracking
        self.round_count = 0
        self.last_selections = {}
        self.recent_scores = []
        self.improvement_history = []

        # Debug
        total_arms = sum(len(model.arms) for model in self.param_models.values())
        print(
            f"BayesianRL: {len(self.param_models)} parameters, {total_arms} total arms"
        )
        print(f"Thompson Sampling with Bayesian Posteriors")
        for param_name, model in self.param_models.items():
            print(
                f"  {param_name}: {len(model.arms)} arms ({model.param_config['type']})"
            )

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Bayesian RL Parameter Suggestion"""
        try:
            self.round_count = round_num

            suggested_params = {}
            selected_arms = {}

            if round_num < self.initial_exploration:
                # Initial Random Exploration
                for param_name, model in self.param_models.items():
                    arm_idx = random.randint(0, len(model.arms) - 1)
                    value = model.sample_from_arm(arm_idx)
                    suggested_params[param_name] = value
                    selected_arms[param_name] = arm_idx

                print(
                    f"Round {round_num}: BAYESIAN INITIAL EXPLORATION ({self.initial_exploration - round_num} left)"
                )

            else:
                # Bayesian Thompson Sampling
                decision_info = []

                for param_name, model in self.param_models.items():
                    try:
                        # Thompson Sampling basierend auf Posterior
                        arm_idx = model.thompson_sampling()

                        # Uncertainty Bonus (mehr Exploration für unsichere Parameter)
                        if self._is_parameter_uncertain(model):
                            if random.random() < self.uncertainty_bonus:
                                # Force exploration für unsichere Parameter
                                arm_idx = random.randint(0, len(model.arms) - 1)
                                decision_info.append(f"{param_name}:UNCERTAINTY")
                            else:
                                decision_info.append(f"{param_name}:THOMPSON")
                        else:
                            decision_info.append(f"{param_name}:THOMPSON")

                        value = model.sample_from_arm(arm_idx)
                        suggested_params[param_name] = value
                        selected_arms[param_name] = arm_idx

                    except Exception as e:
                        # Fallback for individual parameter
                        arm_idx = random.randint(0, len(model.arms) - 1)
                        value = model.sample_from_arm(arm_idx)
                        suggested_params[param_name] = value
                        selected_arms[param_name] = arm_idx
                        decision_info.append(f"{param_name}:FALLBACK")

                print(f"Round {round_num}: BAYESIAN THOMPSON SAMPLING")
                print(f"  Decisions: {', '.join(decision_info)}")

            # Store für Update
            self.last_selections = selected_arms

            print(f"Suggested: {suggested_params}")
            return suggested_params

        except Exception as e:
            print(f"Error in BayesianRL suggest_hyperparameters: {e}")
            # Emergency fallback: pure random
            return {
                param_name: self._random_param_value(param_name, param_config)
                for param_name, param_config in self.hyperparameter_space.items()
            }

    def _random_param_value(self, param_name: str, param_config: Dict):
        """Generate random value for parameter"""
        try:
            if param_config["type"] == "categorical":
                return random.choice(param_config["values"])
            elif param_config["type"] == "int":
                return random.randint(param_config["min"], param_config["max"])
            elif param_config["type"] == "float":
                return random.uniform(param_config["min"], param_config["max"])
        except:
            # Ultimate fallback
            return param_config.get("default", 0)

    def _is_parameter_uncertain(self, model: BayesianParameterModel) -> bool:
        """Prüfe ob Parameter noch unsicher ist (braucht mehr Exploration)"""
        try:
            if model.param_config["type"] == "categorical":
                # Unsicher wenn Beta-Verteilungen noch schwach sind
                total_observations = np.sum(model.alpha + model.beta_param)
                return total_observations < 20  # Weniger als 20 Beobachtungen
            else:
                # Unsicher wenn Varianzen noch hoch sind oder wenige Beobachtungen
                max_variance = np.max(model.variances)
                min_observations = np.min(model.counts)
                return max_variance > 0.5 or min_observations < 5
        except:
            return True  # Default to uncertain if error

    def update(self, hyperparameters: Dict, score: float):
        """Bayesian Posterior Updates"""
        try:
            # Handle NaN scores
            if np.isnan(score):
                score = -1.0
                print(f"Score: NaN (penalized to {score})")

            # Global tracking
            self.recent_scores.append(score)
            self.history.append({"params": hyperparameters, "score": score})

            # Track improvements
            improved = False
            if score > self.best_score:
                improvement = score - self.best_score
                self.improvement_history.append((self.round_count, improvement))
                self.best_score = score
                self.best_params = hyperparameters.copy()
                improved = True
                print(
                    f"  *** BAYESIAN IMPROVEMENT: +{improvement:.4f} -> {score:.4f} ***"
                )

            # Bayesian Posterior Updates für alle Parameter
            if self.round_count >= self.initial_exploration and self.last_selections:
                for param_name, arm_idx in self.last_selections.items():
                    if param_name in self.param_models:
                        model = self.param_models[param_name]
                        model.update_posterior(arm_idx, score)

            print(f"Score: {score:.4f}")

            # Zeige Bayesian Learning Progress
            if (
                self.round_count % 30 == 0
                and self.round_count > self.initial_exploration
            ):
                self._print_bayesian_progress()

        except Exception as e:
            print(f"Error in BayesianRL update: {e}")

    def _print_bayesian_progress(self):
        """Zeige Bayesian Learning Progress"""
        try:
            print(f"\n  === Bayesian Learning Progress (Round {self.round_count}) ===")

            for param_name, model in self.param_models.items():
                try:
                    if len(model.arms) <= 5:  # Nur für überschaubare Parameter
                        print(f"  {param_name} ({model.param_config['type']}):")

                        stats = model.get_posterior_stats()
                        if not stats:
                            continue

                        # Sortiere nach Posterior Mean - SAFE VERSION
                        stats_items = list(stats.items())
                        sorted_arms = sorted(
                            stats_items,
                            key=lambda x: x[1]["posterior_mean"],
                            reverse=True,
                        )

                        # Safe slicing - take only first 3 items
                        top_arms = (
                            sorted_arms[:3] if len(sorted_arms) >= 3 else sorted_arms
                        )

                        for arm_name, arm_stats in top_arms:
                            try:
                                if model.param_config["type"] == "categorical":
                                    mean = arm_stats["posterior_mean"]
                                    confidence = arm_stats["confidence"]
                                    print(
                                        f"    {arm_name}: p={mean:.3f} (conf={confidence:.0f})"
                                    )
                                else:
                                    mean = arm_stats["posterior_mean"]
                                    var = arm_stats["posterior_var"]
                                    obs = arm_stats["observations"]
                                    print(
                                        f"    {arm_name}: μ={mean:.3f}±{np.sqrt(max(0, var)):.3f} (n={obs:.0f})"
                                    )
                            except Exception as e:
                                continue
                except Exception as e:
                    continue

            # Uncertainty status
            try:
                uncertain_params = []
                for param_name, model in self.param_models.items():
                    if self._is_parameter_uncertain(model):
                        uncertain_params.append(param_name)

                if uncertain_params:
                    print(f"  Still uncertain: {uncertain_params}")
                else:
                    print(f"  All parameters converged!")
            except Exception as e:
                pass

            print()

        except Exception as e:
            print(f"  Error in progress reporting: {e}")
            print()

    def get_best_bayesian_prediction(self) -> Dict:
        """Hole die besten Parameter basierend auf Bayesian Posteriors"""
        try:
            best_params = {}

            for param_name, model in self.param_models.items():
                try:
                    if model.param_config["type"] == "categorical":
                        # Wähle Arm mit höchster Posterior-Wahrscheinlichkeit
                        posterior_means = []
                        for i in range(len(model.arms)):
                            alpha_val = float(model.alpha[i])
                            beta_val = float(model.beta_param[i])
                            mean = alpha_val / (alpha_val + beta_val)
                            posterior_means.append(mean)

                        best_arm_idx = int(np.argmax(posterior_means))
                        best_params[param_name] = model.arms[best_arm_idx]
                    else:
                        # Wähle Arm mit höchstem Posterior Mean
                        best_arm_idx = int(np.argmax(model.means))
                        best_params[param_name] = model.sample_from_arm(best_arm_idx)
                except Exception as e:
                    # Fallback for individual parameter
                    best_params[param_name] = self._random_param_value(
                        param_name, model.param_config
                    )

            return best_params

        except Exception as e:
            # Return random parameters as fallback
            return {
                param_name: self._random_param_value(param_name, param_config)
                for param_name, param_config in self.hyperparameter_space.items()
            }
