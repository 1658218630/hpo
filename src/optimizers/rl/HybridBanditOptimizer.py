import numpy as np
import random
from typing import Dict, List, Tuple, Any
from collections import defaultdict, deque
from optimizers.BaseOptimizer import BaseOptimizer
import scipy.stats as stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel


class HybridBanditOptimizer(BaseOptimizer):
    """
    Hybrid Multi-Armed Bandit with Bayesian Optimization (HBMAB)
    """

    def __init__(
        self,
        hyperparameter_space: Dict,
        initial_random_samples: int = 10,
        bandit_c: float = 2.0,  # UCB exploration parameter
        gp_acquisition_weight: float = 0.8,  # Balance exploitation vs exploration
        adaptation_window: int = 20,  # Window for strategy adaptation
        min_improvement_threshold: float = 0.001,  # Early stopping threshold
        stagnation_patience: int = 15,  # Rounds without improvement before strategy change
    ):
        super().__init__("HybridBandit", 1000)

        self.hyperparameter_space = hyperparameter_space
        self.initial_random_samples = initial_random_samples
        self.bandit_c = bandit_c
        self.gp_acquisition_weight = gp_acquisition_weight
        self.adaptation_window = adaptation_window
        self.min_improvement_threshold = min_improvement_threshold
        self.stagnation_patience = stagnation_patience

        # Separate continuous and discrete parameters
        self.continuous_params = {}
        self.discrete_params = {}
        self._categorize_parameters()

        # Multi-Armed Bandit for discrete parameters
        self.discrete_arms = self._create_discrete_arms()
        self.arm_rewards = defaultdict(list)
        self.arm_counts = defaultdict(int)

        # Gaussian Process for continuous parameters
        if self.continuous_params:
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            self.gp = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6, normalize_y=True
            )
            self.gp_X = []
            self.gp_y = []

        # Parameter importance tracking
        self.param_importance = {param: 1.0 for param in hyperparameter_space.keys()}
        self.param_performance_history = defaultdict(list)

        # Strategy management
        self.current_strategy = "random"  # "random", "bandit", "bayesian", "hybrid"
        self.strategy_performance = defaultdict(list)
        self.rounds_since_improvement = 0
        self.performance_history = deque(maxlen=adaptation_window)

        # Best configuration tracking
        self.best_config_score = -float("inf")
        self.best_config_params = None
        self.recent_best_scores = deque(maxlen=5)

        print(
            f"HBMAB: {len(self.discrete_arms)} discrete arms, "
            f"{len(self.continuous_params)} continuous params"
        )

    def _categorize_parameters(self):
        """Separate parameters into continuous and discrete categories"""
        for param_name, param_config in self.hyperparameter_space.items():
            if param_config["type"] == "categorical":
                self.discrete_params[param_name] = param_config
            else:
                self.continuous_params[param_name] = param_config

    def _create_discrete_arms(self):
        """Create arms for discrete parameter combinations"""
        if not self.discrete_params:
            return [{}]

        arms = []
        # Create combinations of discrete parameter values
        param_names = list(self.discrete_params.keys())

        if len(param_names) == 1:
            param = param_names[0]
            for value in self.discrete_params[param]["values"]:
                arms.append({param: value})
        else:
            # For multiple discrete params, create strategic combinations
            # instead of full cartesian product to keep manageable
            for param_name in param_names:
                for value in self.discrete_params[param_name]["values"]:
                    arm = {param_name: value}
                    # Add default values for other discrete params
                    for other_param in param_names:
                        if other_param != param_name:
                            arm[other_param] = self.discrete_params[other_param].get(
                                "default",
                                self.discrete_params[other_param]["values"][0],
                            )
                    arms.append(arm)

        return arms

    def _ucb_select_arm(self) -> Dict:
        """Select discrete parameter combination using Upper Confidence Bound"""
        if not self.discrete_arms:
            return {}

        total_counts = sum(self.arm_counts.values())
        if total_counts == 0:
            return random.choice(self.discrete_arms)

        best_ucb = -float("inf")
        best_arm = None

        for i, arm in enumerate(self.discrete_arms):
            arm_key = str(sorted(arm.items()))

            if self.arm_counts[arm_key] == 0:
                return arm  # Unvisited arm

            mean_reward = np.mean(self.arm_rewards[arm_key])
            confidence = self.bandit_c * np.sqrt(
                np.log(total_counts) / self.arm_counts[arm_key]
            )
            ucb_value = mean_reward + confidence

            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_arm = arm

        return best_arm if best_arm is not None else random.choice(self.discrete_arms)

    def _bayesian_suggest_continuous(self, discrete_config: Dict) -> Dict:
        """Use Gaussian Process to suggest continuous parameters"""
        if not self.continuous_params or len(self.gp_y) < 3:
            return self._random_continuous_params()

        # Fit GP if we have enough data
        if len(self.gp_X) != len(self.gp_y):
            return self._random_continuous_params()

        try:
            self.gp.fit(self.gp_X, self.gp_y)
        except:
            return self._random_continuous_params()

        # Generate candidates and select best according to acquisition function
        best_params = None
        best_acquisition = -float("inf")

        n_candidates = 50
        for _ in range(n_candidates):
            candidate = self._random_continuous_params()
            candidate_vector = self._params_to_vector(candidate)

            try:
                mean, std = self.gp.predict([candidate_vector], return_std=True)
                # Expected Improvement acquisition function
                if self.best_score > -float("inf"):
                    z = (mean[0] - self.best_score) / (std[0] + 1e-9)
                    ei = (mean[0] - self.best_score) * stats.norm.cdf(z) + std[
                        0
                    ] * stats.norm.pdf(z)
                else:
                    ei = std[0]  # Pure exploration

                # Weight by parameter importance
                importance_weight = self._calculate_importance_weight(candidate)
                acquisition = (
                    self.gp_acquisition_weight * ei
                    + (1 - self.gp_acquisition_weight) * importance_weight
                )

                if acquisition > best_acquisition:
                    best_acquisition = acquisition
                    best_params = candidate
            except:
                continue

        return (
            best_params if best_params is not None else self._random_continuous_params()
        )

    def _random_continuous_params(self) -> Dict:
        """Generate random continuous parameters"""
        params = {}
        for param_name, param_config in self.continuous_params.items():
            if param_config["type"] == "int":
                params[param_name] = random.randint(
                    param_config["min"], param_config["max"]
                )
            elif param_config["type"] == "float":
                params[param_name] = random.uniform(
                    param_config["min"], param_config["max"]
                )
        return params

    def _params_to_vector(self, params: Dict) -> List[float]:
        """Convert continuous parameters to vector for GP"""
        vector = []
        for param_name in sorted(self.continuous_params.keys()):
            if param_name in params:
                value = params[param_name]
                param_config = self.continuous_params[param_name]
                # Normalize to [0, 1]
                if param_config["type"] in ["int", "float"]:
                    normalized = (value - param_config["min"]) / (
                        param_config["max"] - param_config["min"]
                    )
                    vector.append(normalized)
        return vector

    def _calculate_importance_weight(self, params: Dict) -> float:
        """Calculate importance weight for parameter combination"""
        weight = 1.0
        for param_name, value in params.items():
            if param_name in self.param_importance:
                weight *= self.param_importance[param_name]
        return weight

    def _update_parameter_importance(self, params: Dict, score: float):
        """Update parameter importance based on performance"""
        for param_name, value in params.items():
            self.param_performance_history[param_name].append(score)

            # Calculate importance as correlation with performance
            if len(self.param_performance_history[param_name]) >= 5:
                scores = self.param_performance_history[param_name][
                    -10:
                ]  # Last 10 scores
                # Simple importance = variance in scores for this parameter
                self.param_importance[param_name] = max(0.1, np.std(scores))

    def _adapt_strategy(self, round_num: int):
        """Adapt optimization strategy based on recent performance"""
        if round_num < self.initial_random_samples:
            self.current_strategy = "random"
            return

        # Check for stagnation
        if len(self.recent_best_scores) >= 3:
            recent_improvement = max(self.recent_best_scores) - min(
                self.recent_best_scores
            )
            if recent_improvement < self.min_improvement_threshold:
                self.rounds_since_improvement += 1
            else:
                self.rounds_since_improvement = 0

        # Strategy selection logic
        if self.rounds_since_improvement > self.stagnation_patience:
            # Switch to more exploratory strategy
            self.current_strategy = "random"
            self.rounds_since_improvement = 0
        elif round_num < 50:
            # Early phase: balanced exploration
            if random.random() < 0.4:
                self.current_strategy = "bandit"
            elif random.random() < 0.7 and self.continuous_params:
                self.current_strategy = "bayesian"
            else:
                self.current_strategy = "hybrid"
        else:
            # Later phase: more exploitation
            if self.continuous_params and len(self.gp_y) >= 10:
                self.current_strategy = "hybrid"
            else:
                self.current_strategy = "bandit"

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Main method to suggest hyperparameters"""
        self._adapt_strategy(round_num)

        # Generate suggestion based on current strategy
        if self.current_strategy == "random" or round_num < 3:
            params = self._generate_random_params()
        elif self.current_strategy == "bandit":
            discrete_config = self._ucb_select_arm()
            continuous_config = self._random_continuous_params()
            params = {**discrete_config, **continuous_config}
        elif self.current_strategy == "bayesian":
            discrete_config = (
                random.choice(self.discrete_arms) if self.discrete_arms else {}
            )
            continuous_config = self._bayesian_suggest_continuous(discrete_config)
            params = {**discrete_config, **continuous_config}
        else:  # hybrid
            discrete_config = self._ucb_select_arm()
            continuous_config = self._bayesian_suggest_continuous(discrete_config)
            params = {**discrete_config, **continuous_config}

        return params

    def _generate_random_params(self) -> Dict:
        """Generate completely random parameters"""
        params = {}
        for param_name, param_config in self.hyperparameter_space.items():
            if param_config["type"] == "int":
                params[param_name] = random.randint(
                    param_config["min"], param_config["max"]
                )
            elif param_config["type"] == "float":
                params[param_name] = random.uniform(
                    param_config["min"], param_config["max"]
                )
            elif param_config["type"] == "categorical":
                params[param_name] = random.choice(param_config["values"])
        return params

    def update(self, hyperparameters: Dict, score: float):
        """Update optimizer with new results"""
        # Base class update
        self.history.append({"params": hyperparameters, "score": score})

        # Track best configuration
        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters.copy()
            self.best_config_score = score
            self.best_config_params = hyperparameters.copy()

        # Update performance tracking
        self.performance_history.append(score)
        self.recent_best_scores.append(score)

        # Update bandit arms
        discrete_part = {
            k: v for k, v in hyperparameters.items() if k in self.discrete_params
        }
        if discrete_part or not self.discrete_params:
            arm_key = str(sorted(discrete_part.items()))
            self.arm_rewards[arm_key].append(score)
            self.arm_counts[arm_key] += 1

        # Update Gaussian Process
        if self.continuous_params:
            continuous_part = {
                k: v for k, v in hyperparameters.items() if k in self.continuous_params
            }
            if continuous_part:
                vector = self._params_to_vector(continuous_part)
                if len(vector) > 0:
                    self.gp_X.append(vector)
                    self.gp_y.append(score)

        # Update parameter importance
        self._update_parameter_importance(hyperparameters, score)

        # Track strategy performance
        self.strategy_performance[self.current_strategy].append(score)
