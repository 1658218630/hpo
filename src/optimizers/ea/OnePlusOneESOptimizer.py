import numpy as np
from typing import Dict, List, Any
from optimizers.BaseOptimizer import BaseOptimizer


class OnePlusOneESOptimizer(BaseOptimizer):
    """
    An optimizer based on the simplest (1+1) Evolution Strategy.

    This strategy maintains a single parent, creates one offspring via mutation,
    and replaces the parent if the offspring is better. The mutation strength (sigma)
    is adapted based on the success of the mutation.
    """

    def __init__(self,
                 hyperparameter_space: Dict[str, Dict],
                 initial_sigma: float = 0.5,
                 success_factor: float = 1.2,  # Factor to increase sigma on success
                 failure_factor: float = 0.85, # Factor to decrease sigma on failure
                 max_evaluations: int = 500):
        """
        Initializes the (1+1) Evolution Strategy Optimizer.
        """
        if not isinstance(hyperparameter_space, dict) or not hyperparameter_space:
            raise ValueError("hyperparameter_space must be a non-empty dictionary.")

        self.param_names: List[str] = list(hyperparameter_space.keys())
        self.param_configs: List[Dict] = [hyperparameter_space[name] for name in self.param_names]
        self.dimensions = len(self.param_names)

        super().__init__(name="OnePlusOneESOptimizer", hyperparameter_count=self.dimensions)

        self.sigma = initial_sigma
        self.success_factor = success_factor
        self.failure_factor = failure_factor
        self.max_evaluations = max_evaluations

        # Extract bounds for each parameter
        self.bounds = np.array([
            (c['min'], c['max']) if c['type'] in ['int', 'float']
            else (0, len(c['values']) - 1)
            for c in self.param_configs
        ]).T
        self.min_bounds, self.max_bounds = self.bounds[0], self.bounds[1]

        # ES state
        self.parent_vector: np.ndarray | None = None
        self.parent_score: float = -np.inf
        self.last_suggested_vector: np.ndarray | None = None
        self.eval_count = 0

    def _decode_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        """Decodes a real-valued vector into a dictionary of hyperparameters."""
        params = {}
        for i, value in enumerate(vector):
            name = self.param_names[i]
            config = self.param_configs[i]
            min_b, max_b = self.bounds.T[i]
            value = np.clip(value, min_b, max_b)

            if config['type'] == 'int':
                params[name] = int(round(value))
            elif config['type'] == 'float':
                params[name] = float(value)
            elif config['type'] == 'categorical':
                idx = int(round(value))
                params[name] = config['values'][idx]
        return params

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Suggests the next set of hyperparameters to evaluate."""
        if self.eval_count >= self.max_evaluations:
            print("Max evaluations reached. Returning best found parameters.")
            return self.get_best_params() if self.best_params else {}

        if self.parent_vector is None:
            # First call: create and suggest the initial parent
            new_vector = np.random.uniform(self.min_bounds, self.max_bounds, self.dimensions)
            print("Suggesting initial parent for (1+1)-ES.")
        else:
            # Subsequent calls: create an offspring by mutating the parent
            mutation = np.random.normal(loc=0.0, scale=self.sigma, size=self.dimensions)
            new_vector = self.parent_vector + mutation
            new_vector = np.clip(new_vector, self.min_bounds, self.max_bounds)
            print(f"Round {self.eval_count + 1}, Suggesting offspring (sigma: {self.sigma:.4f}).")

        self.last_suggested_vector = new_vector
        return self._decode_vector(new_vector)

    def update(self, hyperparameters: Dict, score: float) -> None:
        """Updates the optimizer with the fitness score of the last suggestion."""
        if self.last_suggested_vector is None:
            return

        self.eval_count += 1

        # Update history and overall best
        self.history.append({'params': hyperparameters, 'score': score})
        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters

        # (1+1) selection logic
        if self.parent_vector is None:
            # This was the first evaluation, so it becomes the initial parent
            self.parent_vector = self.last_suggested_vector
            self.parent_score = score
            print(f"Initialized parent with score: {score:.4f}")
        elif score >= self.parent_score:
            # Offspring is better or equal, it becomes the new parent
            self.parent_vector = self.last_suggested_vector
            self.parent_score = score
            self.sigma *= self.success_factor # Increase sigma (exploration)
            print(f"Successful mutation. New parent score: {score:.4f}. Increasing sigma.")
        else:
            # Offspring is worse, parent remains.
            self.sigma *= self.failure_factor # Decrease sigma (exploitation)
            print(f"Unsuccessful mutation. Parent score remains {self.parent_score:.4f}. Decreasing sigma.")

        print(f"Overall best score: {self.get_best_score():.4f}")