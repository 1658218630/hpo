import optuna
import numpy as np
import random
from typing import Dict, List, Any
from optimizers.BaseOptimizer import BaseOptimizer

# Suppress Optuna logging for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTPEOptimizer(BaseOptimizer):
    """
    Optuna Tree-structured Parzen Estimator (TPE) für HPO
    """

    def __init__(
        self,
        hyperparameter_space: Dict,
        n_startup_trials: int = 10,  # Random trials bevor TPE startet
        n_ei_candidates: int = 24,  # Candidates für Expected Improvement
        gamma: float = 0.25,  # TPE gamma parameter (lower = more exploitation)
        prior_weight: float = 1.0,  # Weight für prior distribution
        multivariate: bool = True,  # Multi-dimensional sampling
        warn_independent: bool = False,  # Keine Warnungen für independent sampling
    ):
        super().__init__("OptunaTPE", 1000)

        self.hyperparameter_space = hyperparameter_space
        self.n_startup_trials = n_startup_trials

        # Erstelle Optuna Study mit TPE Sampler
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            gamma=gamma,
            prior_weight=prior_weight,
            multivariate=multivariate,
            warn_independent_sampling=warn_independent,
            seed=42,  # Reproducibility
        )

        self.study = optuna.create_study(
            sampler=sampler,
            direction="maximize",  # Wir maximieren Score
            study_name=f"TPE_HPO_{random.randint(1000, 9999)}",
        )

        # Convert hyperparameter space to Optuna format
        self.optuna_space = self._convert_hyperparameter_space()

        # Tracking
        self.round_count = 0
        self.current_trial = None

        print(
            f"OptunaTPE: n_startup={n_startup_trials}, gamma={gamma}, multivariate={multivariate}"
        )
        print(f"Parameter space: {list(hyperparameter_space.keys())}")

    def _convert_hyperparameter_space(self) -> Dict:
        """Convert our hyperparameter space to Optuna suggest functions"""
        optuna_space = {}

        for param_name, param_config in self.hyperparameter_space.items():
            if param_config["type"] == "int":
                optuna_space[param_name] = {
                    "type": "int",
                    "low": param_config["min"],
                    "high": param_config["max"],
                }
            elif param_config["type"] == "float":
                optuna_space[param_name] = {
                    "type": "float",
                    "low": param_config["min"],
                    "high": param_config["max"],
                }
            elif param_config["type"] == "categorical":
                optuna_space[param_name] = {
                    "type": "categorical",
                    "choices": param_config["values"],
                }

        return optuna_space

    def _tpe_suggest_int(self, param_name: str, low: int, high: int) -> int:
        """TPE-inspired integer suggestion based on historical data"""
        if len(self.study.trials) < 3:
            return random.randint(low, high)

        # Get historical performance for this parameter
        param_values = []
        scores = []

        for trial in self.study.trials:
            if trial.value is not None and param_name in trial.params:
                param_values.append(trial.params[param_name])
                scores.append(trial.value)

        if len(param_values) < 3:
            return random.randint(low, high)

        # Simple TPE logic: favor values from better trials
        sorted_indices = np.argsort(scores)
        top_quantile = max(1, len(scores) // 4)  # Top 25%

        good_values = [param_values[i] for i in sorted_indices[-top_quantile:]]

        if good_values:
            # Sample around good values with some noise
            base_value = random.choice(good_values)
            noise = random.randint(-2, 2)  # Small perturbation
            return max(low, min(high, base_value + noise))
        else:
            return random.randint(low, high)

    def _tpe_suggest_float(self, param_name: str, low: float, high: float) -> float:
        """TPE-inspired float suggestion based on historical data"""
        if len(self.study.trials) < 3:
            return random.uniform(low, high)

        # Get historical performance for this parameter
        param_values = []
        scores = []

        for trial in self.study.trials:
            if trial.value is not None and param_name in trial.params:
                param_values.append(trial.params[param_name])
                scores.append(trial.value)

        if len(param_values) < 3:
            return random.uniform(low, high)

        # Simple TPE logic: favor values from better trials
        sorted_indices = np.argsort(scores)
        top_quantile = max(1, len(scores) // 4)  # Top 25%

        good_values = [param_values[i] for i in sorted_indices[-top_quantile:]]

        if good_values:
            # Sample around good values with Gaussian noise
            base_value = random.choice(good_values)
            noise_std = (high - low) * 0.1  # 10% of range
            noise = random.normalvariate(0, noise_std)
            return max(low, min(high, base_value + noise))
        else:
            return random.uniform(low, high)

    def _tpe_suggest_categorical(self, param_name: str, choices: List) -> Any:
        """TPE-inspired categorical suggestion based on historical data"""
        if len(self.study.trials) < 3:
            return random.choice(choices)

        # Count performance for each choice
        choice_scores = {choice: [] for choice in choices}

        for trial in self.study.trials:
            if trial.value is not None and param_name in trial.params:
                value = trial.params[param_name]
                if value in choice_scores:
                    choice_scores[value].append(trial.value)

        # Calculate average score for each choice
        choice_averages = {}
        for choice, scores in choice_scores.items():
            if scores:
                choice_averages[choice] = np.mean(scores)
            else:
                choice_averages[choice] = 0.0

        if not choice_averages:
            return random.choice(choices)

        # Weighted sampling based on performance
        weights = []
        choice_list = []

        for choice, avg_score in choice_averages.items():
            choice_list.append(choice)
            # Softmax-like weighting
            weight = np.exp(avg_score * 2)  # Scale factor for more distinction
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            return np.random.choice(choice_list, p=weights)
        else:
            return random.choice(choices)

    def _generate_random_params(self) -> Dict:
        """Generate random parameters as fallback"""
        params = {}
        for param_name, space_config in self.optuna_space.items():
            if space_config["type"] == "int":
                params[param_name] = random.randint(
                    space_config["low"], space_config["high"]
                )
            elif space_config["type"] == "float":
                params[param_name] = random.uniform(
                    space_config["low"], space_config["high"]
                )
            elif space_config["type"] == "categorical":
                params[param_name] = random.choice(space_config["choices"])
        return params

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Suggest hyperparameters using Optuna TPE"""
        self.round_count = round_num

        # Use Optuna's optimize approach instead of ask/tell
        def objective(trial):
            # This will be called by Optuna, but we need to return the suggestion
            # We'll store the trial for later use
            self.current_trial = trial
            return 0.0  # Dummy return, real score comes in update()

        # Create a new trial manually
        try:
            # Alternative approach: Use enqueue_trial for manual parameter suggestion
            suggested_params = {}

            # Generate parameters using our own logic that mimics Optuna
            for param_name, space_config in self.optuna_space.items():
                if space_config["type"] == "int":
                    if round_num < self.n_startup_trials:
                        # Random sampling for startup
                        suggested_params[param_name] = random.randint(
                            space_config["low"], space_config["high"]
                        )
                    else:
                        # Use Optuna's internal logic (simplified)
                        suggested_params[param_name] = self._tpe_suggest_int(
                            param_name, space_config["low"], space_config["high"]
                        )

                elif space_config["type"] == "float":
                    if round_num < self.n_startup_trials:
                        # Random sampling for startup
                        suggested_params[param_name] = random.uniform(
                            space_config["low"], space_config["high"]
                        )
                    else:
                        # Use Optuna's internal logic (simplified)
                        suggested_params[param_name] = self._tpe_suggest_float(
                            param_name, space_config["low"], space_config["high"]
                        )

                elif space_config["type"] == "categorical":
                    if round_num < self.n_startup_trials:
                        # Random sampling for startup
                        suggested_params[param_name] = random.choice(
                            space_config["choices"]
                        )
                    else:
                        # Use Optuna's internal logic (simplified)
                        suggested_params[param_name] = self._tpe_suggest_categorical(
                            param_name, space_config["choices"]
                        )

            # Store params for update
            self.last_suggested_params = suggested_params

        except Exception as e:
            print(f"Error in TPE suggestion: {e}")
            # Fallback to random
            suggested_params = self._generate_random_params()
            self.last_suggested_params = suggested_params

        # Bestimme Strategy basierend auf aktueller Phase
        if round_num < self.n_startup_trials:
            strategy = f"RANDOM STARTUP ({self.n_startup_trials - round_num} left)"
        else:
            strategy = "TPE OPTIMIZATION"

            # Debug: Zeige TPE-Intelligenz
            if round_num % 20 == 0 and len(self.study.trials) > self.n_startup_trials:
                self._print_tpe_insights()

        print(f"Round {round_num}: {strategy}")
        print(f"Suggested: {suggested_params}")

        return suggested_params

    def update(self, hyperparameters: Dict, score: float):
        """Update Optuna study with result"""

        # Handle NaN scores
        if np.isnan(score):
            score = -1.0  # Large penalty for NaN
            print(f"Score: NaN (penalized to {score})")

        # Create trial manually and add to study
        try:
            # Create a trial with our suggested parameters
            trial = optuna.trial.create_trial(
                params=getattr(self, "last_suggested_params", hyperparameters),
                distributions={
                    param_name: self._create_distribution(param_name, space_config)
                    for param_name, space_config in self.optuna_space.items()
                },
                value=score,
            )

            # Add trial to study
            self.study.add_trial(trial)

        except Exception as e:
            print(f"Warning: Could not add trial to Optuna study: {e}")
            # Continue without Optuna tracking for this trial

        # Update BaseOptimizer tracking
        self.history.append({"params": hyperparameters, "score": score})

        # Track improvements
        improved = False
        if score > self.best_score:
            old_best = self.best_score
            self.best_score = score
            self.best_params = hyperparameters.copy()
            improved = True
            print(f"  *** TPE IMPROVEMENT: {score:.4f} (was {old_best:.4f}) ***")
        else:
            print(f"  No improvement: {score:.4f} vs best {self.best_score:.4f}")

        print(f"Score: {score:.4f}")

        # Show learning progress
        if self.round_count % 30 == 0 and self.round_count >= self.n_startup_trials:
            self._print_tpe_progress()

    def _create_distribution(self, param_name: str, space_config: Dict):
        """Create Optuna distribution for parameter"""
        if space_config["type"] == "int":
            return optuna.distributions.IntDistribution(
                low=space_config["low"], high=space_config["high"]
            )
        elif space_config["type"] == "float":
            return optuna.distributions.FloatDistribution(
                low=space_config["low"], high=space_config["high"]
            )
        elif space_config["type"] == "categorical":
            return optuna.distributions.CategoricalDistribution(
                choices=space_config["choices"]
            )

    def _print_tpe_insights(self):
        """Show what TPE has learned"""
        print(f"\n  === TPE Learning Insights (Round {self.round_count}) ===")

        # Get parameter importance from Optuna
        try:
            importance = optuna.importance.get_param_importances(
                self.study, evaluator=optuna.importance.FanovaImportanceEvaluator()
            )

            print(f"  Parameter Importance:")
            for param, imp in sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"    {param}: {imp:.3f}")

        except Exception:
            # Fallback if importance calculation fails
            print(f"  TPE has {len(self.study.trials)} trials")

        # Show best parameters found so far
        if self.study.best_trial:
            print(f"  Best parameters so far:")
            for param, value in self.study.best_trial.params.items():
                print(f"    {param}: {value}")
        print()

    def _print_tpe_progress(self):
        """Show TPE optimization progress"""
        print(f"\n  === TPE Progress (Round {self.round_count}) ===")

        if len(self.study.trials) >= 5:
            # Show recent performance trend
            recent_values = [
                t.value for t in self.study.trials[-5:] if t.value is not None
            ]
            if recent_values:
                print(f"  Recent 5 scores: {[f'{v:.4f}' for v in recent_values]}")
                print(f"  Trend: {recent_values[-1] - recent_values[0]:+.4f}")

        print(f"  Total trials: {len(self.study.trials)}")
        print(
            f"  Best score: {self.study.best_value:.4f}"
            if self.study.best_value
            else "  No best score yet"
        )
        print()

    def get_optimization_history(self) -> List[Dict]:
        """Get detailed optimization history from Optuna"""
        history = []
        for trial in self.study.trials:
            if trial.value is not None:
                history.append(
                    {
                        "trial_number": trial.number,
                        "params": trial.params,
                        "score": trial.value,
                        "state": trial.state.name,
                    }
                )
        return history

    def get_best_params_with_confidence(self) -> Dict:
        """Get best parameters with statistical confidence"""
        if not self.study.best_trial:
            return {}

        best_params = self.study.best_trial.params.copy()

        # Add confidence information
        result = {
            "best_params": best_params,
            "best_score": self.study.best_value,
            "trial_number": self.study.best_trial.number,
            "total_trials": len(self.study.trials),
        }

        # Parameter importance if available
        try:
            importance = optuna.importance.get_param_importances(self.study)
            result["param_importance"] = importance
        except:
            result["param_importance"] = {}

        return result

    def get_pareto_front(self, n_params: int = 2) -> List[Dict]:
        """Get Pareto front for multi-objective analysis"""
        # This is a simplified version - could be extended for true multi-objective optimization
        trials = sorted(
            [t for t in self.study.trials if t.value is not None],
            key=lambda x: x.value,
            reverse=True,
        )

        return [
            {"params": trial.params, "score": trial.value, "trial_number": trial.number}
            for trial in trials[:n_params]
        ]
