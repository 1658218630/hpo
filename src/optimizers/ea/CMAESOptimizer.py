import numpy as np
from typing import Dict, List, Any
from optimizers.BaseOptimizer import BaseOptimizer


class CMAESOptimizer(BaseOptimizer):
    """
    An optimizer based on the Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    """

    def __init__(self,
                 hyperparameter_space: Dict[str, Dict],
                 population_size: int = None,
                 initial_sigma: float = 0.3,
                 max_generations: int = 100):
        """
        Initializes the CMA-ES Optimizer.
        """
        if not isinstance(hyperparameter_space, dict) or not hyperparameter_space:
            raise ValueError("hyperparameter_space must be a non-empty dictionary.")

        self.param_names: List[str] = list(hyperparameter_space.keys())
        self.param_configs: List[Dict] = [hyperparameter_space[name] for name in self.param_names]
        self.dimensions = len(self.param_names)

        super().__init__(name="CMAESOptimizer", hyperparameter_count=self.dimensions)

        # CMA-ES parameters
        self.sigma = initial_sigma
        self.max_generations = max_generations

        # Set population size (lambda) based on dimensionality if not provided
        self.population_size = population_size or (4 + int(3 * np.log(self.dimensions)))

        # Extract bounds for each parameter
        self.bounds = np.array([
            (c['min'], c['max']) if c['type'] in ['int', 'float']
            else (0, len(c['values']) - 1)
            for c in self.param_configs
        ]).T
        self.min_bounds, self.max_bounds = self.bounds[0], self.bounds[1]

        # CMA-ES state variables
        self.mean: np.ndarray | None = None
        self.C: np.ndarray | None = None
        self.p_sigma: np.ndarray | None = None
        self.p_c: np.ndarray | None = None

        self.offspring: np.ndarray | None = None
        self.offspring_fitness: List[float] = []

        self.current_generation_num = 0
        self.eval_idx = 0

        self._initialize_state()

    def _initialize_state(self):
        """Initializes the CMA-ES state variables."""
        # 1. Initialize mean randomly within bounds
        self.mean = np.random.uniform(self.min_bounds, self.max_bounds, self.dimensions)

        # 2. Initialize evolution paths and covariance matrix
        self.p_sigma = np.zeros(self.dimensions)
        self.p_c = np.zeros(self.dimensions)
        self.C = np.eye(self.dimensions)

        # 3. Set strategy parameters (weights, learning rates)
        self.mu = self.population_size // 2

        # Recombination weights
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mu_eff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)

        # Learning rates for covariance matrix and step-size
        self.c_c = (4 + self.mu_eff / self.dimensions) / (self.dimensions + 4 + 2 * self.mu_eff / self.dimensions)
        self.c_sigma = (self.mu_eff + 2) / (self.dimensions + self.mu_eff + 5)
        self.c_1 = 2 / ((self.dimensions + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1,
                        2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dimensions + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dimensions + 1)) - 1) + self.c_sigma

        self.chiN = self.dimensions ** 0.5 * (1 - 1 / (4 * self.dimensions) + 1 / (21 * self.dimensions ** 2))

        self.offspring = np.zeros((self.population_size, self.dimensions))
        self.offspring_fitness = []
        self.eval_idx = 0
        self.current_generation_num = 0
        print("Initialized CMA-ES state.")

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

    def _evolve(self):
        """Updates the distribution mean, step-size, and covariance matrix."""
        print(f"--- Evolving CMA-ES for Generation {self.current_generation_num + 1} ---")

        # 1. Sort offspring by fitness (descending)
        sorted_indices = np.argsort(self.offspring_fitness)[::-1]
        best_offspring = self.offspring[sorted_indices[:self.mu]]

        # 2. Update mean
        old_mean = self.mean
        self.mean = self.weights @ best_offspring

        # 3. Update evolution paths
        y_w = (self.mean - old_mean) / self.sigma
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
                       np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * np.linalg.solve(
            np.linalg.cholesky(self.C), y_w)

        h_sigma = int(np.linalg.norm(self.p_sigma) / np.sqrt(
            1 - (1 - self.c_sigma) ** (2 * (self.current_generation_num + 1))) < (
                                  1.4 + 2 / (self.dimensions + 1)) * self.chiN)

        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w

        # 4. Update covariance matrix C
        artmp = (best_offspring - old_mean) / self.sigma
        self.C = (1 - self.c_1 - self.c_mu) * self.C + \
                 self.c_1 * (np.outer(self.p_c, self.p_c) + (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.C) + \
                 self.c_mu * (artmp.T @ np.diag(self.weights) @ artmp)

        # 5. Update step-size sigma
        self.sigma *= np.exp((self.c_sigma / self.damps) * (np.linalg.norm(self.p_sigma) / self.chiN - 1))

        # Reset for next generation
        self.offspring_fitness = []
        self.eval_idx = 0
        self.current_generation_num += 1

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Suggests the next set of hyperparameters to evaluate."""
        if self.eval_idx >= self.population_size:
            if self.current_generation_num < self.max_generations - 1:
                self._evolve()
            else:
                print("Max generations reached. Returning best found parameters.")
                return self.get_best_params() if self.best_params else {}

        # Sample a new individual from the multivariate normal distribution
        try:
            C_decomposed = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            # If C is not positive definite, perform eigendecomposition and repair
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals[eigvals < 0] = 1e-12
            self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_decomposed = np.linalg.cholesky(self.C)

        sample = np.random.randn(self.dimensions)
        new_offspring = self.mean + self.sigma * C_decomposed @ sample

        # Store the generated offspring (vector)
        self.offspring[self.eval_idx] = new_offspring

        print(f"Gen {self.current_generation_num}, Suggesting offspring {self.eval_idx + 1}/{self.population_size}")
        return self._decode_vector(new_offspring)

    def update(self, hyperparameters: Dict, score: float) -> None:
        """Updates the optimizer with the fitness score of an offspring."""
        self.offspring_fitness.append(score)

        self.history.append({'params': hyperparameters, 'score': score})
        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters

        print(f"Gen {self.current_generation_num}, Updated offspring {self.eval_idx + 1}/{self.population_size} "
              f"with score: {score:.4f}. Overall best: {self.get_best_score():.4f}")

        self.eval_idx += 1