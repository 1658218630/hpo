import numpy as np
import random
from typing import Dict, List, Any
from optimizers.BaseOptimizer import BaseOptimizer


class MuCommaLambdaESOptimizer(BaseOptimizer):
    """
    An optimizer based on a (μ,λ) Evolution Strategy (ES).

    In this strategy, μ parents generate λ offspring. The next generation of
    parents is selected *only* from the best individuals of the λ offspring.
    The parent population is completely discarded, which can help escape local optima.
    This requires that λ (offspring_size) must be greater than or equal to μ (population_size).
    """

    def __init__(self,
                 hyperparameter_space: Dict[str, Dict],
                 population_size: int = 10,  # Number of parents (μ)
                 offspring_size: int = 40,   # Number of offspring to generate (λ)
                 initial_sigma: float = 0.5, # Initial mutation strength
                 learning_rate: float = 0.98, # Rate to adapt sigma (closer to 1 is slower)
                 max_generations: int = 100):
        """
        Initializes the (μ,λ) Evolution Strategy Optimizer.
        """
        if not isinstance(hyperparameter_space, dict) or not hyperparameter_space:
            raise ValueError("hyperparameter_space must be a non-empty dictionary.")
        if offspring_size < population_size:
            raise ValueError("For (μ,λ)-ES, offspring_size (λ) must be >= population_size (μ).")

        self.param_names: List[str] = list(hyperparameter_space.keys())
        self.param_configs: List[Dict] = [hyperparameter_space[name] for name in self.param_names]
        self.dimensions = len(self.param_names)

        super().__init__(name="MuCommaLambdaESOptimizer", hyperparameter_count=self.dimensions)

        self.population_size = population_size # μ
        self.offspring_size = offspring_size   # λ
        self.sigma = initial_sigma
        self.learning_rate = learning_rate
        self.max_generations = max_generations

        # Extract bounds for each parameter
        self.bounds = np.array([
            (c['min'], c['max']) if c['type'] in ['int', 'float']
            else (0, len(c['values']) - 1)
            for c in self.param_configs
        ]).T
        self.min_bounds, self.max_bounds = self.bounds[0], self.bounds[1]

        # ES state
        self.population: np.ndarray | None = None # Current parents
        self.offspring: np.ndarray | None = None # Generated offspring
        self.offspring_fitness: List[float] = []

        self.current_generation_num = 0
        self.eval_idx = 0 # Index of the offspring to evaluate next

        self._initialize_population()

    def _initialize_population(self):
        """Initializes the parent population."""
        self.population = np.random.uniform(
            low=self.min_bounds, high=self.max_bounds, size=(self.population_size, self.dimensions)
        )
        self.offspring = np.zeros((self.offspring_size, self.dimensions))
        self.offspring_fitness = []
        self.eval_idx = 0
        self.current_generation_num = 0
        print("Initialized (μ,λ)-ES population.")

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
        """Selects the next generation of parents from the offspring."""
        print(f"--- Evolving to Generation {self.current_generation_num + 1} ---")

        # Sort offspring by fitness (descending)
        sorted_indices = np.argsort(self.offspring_fitness)[::-1]

        # Select the top μ individuals from the λ offspring to be the new parents
        best_offspring_indices = sorted_indices[:self.population_size]
        self.population = self.offspring[best_offspring_indices]

        # Adapt sigma using a simple decay factor
        self.sigma *= self.learning_rate

        # Reset for the next generation
        self.offspring = np.zeros((self.offspring_size, self.dimensions))
        self.offspring_fitness = []
        self.eval_idx = 0
        self.current_generation_num += 1

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Suggests the next set of hyperparameters to evaluate."""
        if self.eval_idx >= self.offspring_size:
            if self.current_generation_num < self.max_generations - 1:
                self._evolve()
            else:
                print("Max generations reached. Returning best found parameters.")
                return self.get_best_params() if self.best_params else {}

        # Select a parent randomly from the population
        parent = self.population[random.randint(0, self.population_size - 1)]

        # Create a new offspring by mutating the parent
        mutation = np.random.normal(loc=0.0, scale=self.sigma, size=self.dimensions)
        new_offspring = parent + mutation
        new_offspring = np.clip(new_offspring, self.min_bounds, self.max_bounds)

        # Store the generated offspring
        self.offspring[self.eval_idx] = new_offspring

        print(f"Gen {self.current_generation_num}, Suggesting offspring {self.eval_idx + 1}/{self.offspring_size}")
        return self._decode_vector(new_offspring)

    def update(self, hyperparameters: Dict, score: float) -> None:
        """Updates the optimizer with the fitness score of an offspring."""
        self.offspring_fitness.append(score)

        self.history.append({'params': hyperparameters, 'score': score})
        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters

        print(f"Gen {self.current_generation_num}, Updated offspring {self.eval_idx + 1}/{self.offspring_size} "
              f"with score: {score:.4f}. Overall best: {self.get_best_score():.4f}")

        self.eval_idx += 1