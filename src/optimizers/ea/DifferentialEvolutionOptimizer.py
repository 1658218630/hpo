import random
import numpy as np
from typing import Dict, List, Any
from optimizers.BaseOptimizer import BaseOptimizer


class DifferentialEvolutionOptimizer(BaseOptimizer):
    """
    An optimizer based on the Differential Evolution (DE) algorithm.
    """

    def __init__(self,
                 hyperparameter_space: Dict[str, Dict],
                 population_size: int = 50,
                 f_weight: float = 0.8,  # Weighting factor for mutation (F)
                 cr_prob: float = 0.9,  # Crossover probability (CR)
                 max_generations: int = 100):
        """
        Initializes the Differential Evolution Optimizer.
        """
        if not isinstance(hyperparameter_space, dict) or not hyperparameter_space:
            raise ValueError("hyperparameter_space must be a non-empty dictionary.")

        self.param_names: List[str] = list(hyperparameter_space.keys())
        self.param_configs: List[Dict] = [hyperparameter_space[name] for name in self.param_names]
        self.chromosome_length = len(self.param_names)

        super().__init__(name="DifferentialEvolutionOptimizer", hyperparameter_count=self.chromosome_length)

        self.population_size = population_size
        self.f_weight = f_weight
        self.cr_prob = cr_prob
        self.max_generations = max_generations

        # Extract bounds for each parameter for real-valued vectors
        self.bounds = []
        for config in self.param_configs:
            if config['type'] in ['int', 'float']:
                self.bounds.append((config['min'], config['max']))
            else:  # Categorical
                # For simplicity, we map categories to an integer range [0, num_values-1]
                self.bounds.append((0, len(config['values']) - 1))

        self.population: np.ndarray | None = None
        self.fitness: List[float] = []
        self.current_generation_num = 0
        self.eval_idx = 0  # Index of the individual to evaluate next

        self._initialize_population()

    def _initialize_population(self):
        """Initializes the population with random real-valued vectors within bounds."""
        self.population = np.zeros((self.population_size, self.chromosome_length))
        for i in range(self.population_size):
            for j in range(self.chromosome_length):
                min_b, max_b = self.bounds[j]
                self.population[i, j] = random.uniform(min_b, max_b)
        self.fitness = [-np.inf] * self.population_size
        self.eval_idx = 0
        self.current_generation_num = 0
        print("Initialized Differential Evolution population.")

    def _decode_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        """Decodes a real-valued vector into a dictionary of hyperparameters."""
        params = {}
        for i, value in enumerate(vector):
            name = self.param_names[i]
            config = self.param_configs[i]
            min_b, max_b = self.bounds[i]

            # Ensure value is within bounds
            value = np.clip(value, min_b, max_b)

            if config['type'] == 'int':
                params[name] = int(round(value))
            elif config['type'] == 'float':
                params[name] = float(value)
            elif config['type'] == 'categorical':
                # Round to nearest index and select from values list
                idx = int(round(value))
                params[name] = config['values'][idx]
        return params

    def _evolve(self):
        """
        Creates the next generation of the population using the DE/rand/1/bin scheme.
        """
        print(f"--- Evolving to Generation {self.current_generation_num + 1} ---")

        # Extract min and max bounds for efficient clipping
        min_bounds = np.array([b[0] for b in self.bounds])
        max_bounds = np.array([b[1] for b in self.bounds])

        new_population = np.zeros_like(self.population)

        for i in range(self.population_size):
            # 1. Select three distinct individuals (a, b, c) that are not the current individual
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[random.sample(idxs, 3)]

            # 2. Create a mutant vector using the differential weight
            mutant = a + self.f_weight * (b - c)
            # Ensure the mutant vector stays within the parameter bounds
            mutant = np.clip(mutant, min_bounds, max_bounds)

            # 3. Create a trial vector via binomial crossover
            trial = np.copy(self.population[i])
            cross_points = np.random.rand(self.chromosome_length) < self.cr_prob

            # Ensure at least one parameter from the mutant is included
            if not np.any(cross_points):
                cross_points[random.randrange(self.chromosome_length)] = True

            trial[cross_points] = mutant[cross_points]

            # The trial vector becomes the new individual for the next generation
            new_population[i] = trial

        # Replace the old population with the newly generated one
        self.population = new_population

        # Reset state for the new generation
        self.fitness = [-np.inf] * self.population_size
        self.eval_idx = 0
        self.current_generation_num += 1

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Suggests the next set of hyperparameters to evaluate."""
        if self.eval_idx >= self.population_size:
            # A full generation has been evaluated, time to evolve
            if self.current_generation_num < self.max_generations - 1:
                self._evolve()
            else:
                print("Max generations reached. Returning best found parameters.")
                return self.get_best_params() if self.best_params else {}

        # Suggest the next individual in the current population
        individual_vector = self.population[self.eval_idx]

        print(f"Gen {self.current_generation_num}, Suggesting individual {self.eval_idx + 1}/{self.population_size}")
        return self._decode_vector(individual_vector)

    def update(self, hyperparameters: Dict, score: float) -> None:
        """Updates the optimizer with the fitness score."""
        # Update the fitness of the evaluated individual
        self.fitness[self.eval_idx] = score

        # Update BaseOptimizer state
        self.history.append({'params': hyperparameters, 'score': score})
        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters

        print(f"Gen {self.current_generation_num}, Updated individual {self.eval_idx + 1}/{self.population_size} "
              f"with score: {score:.4f}. Overall best: {self.get_best_score():.4f}")

        # Move to the next individual
        self.eval_idx += 1