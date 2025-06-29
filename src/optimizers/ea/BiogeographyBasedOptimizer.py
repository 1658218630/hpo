import numpy as np
import random
from typing import Dict, List, Any
from optimizers.BaseOptimizer import BaseOptimizer


class BiogeographyBasedOptimizer(BaseOptimizer):
    """
    An optimizer based on the Biogeography-Based Optimization (BBO) algorithm.
    """

    def __init__(self,
                 hyperparameter_space: Dict[str, Dict],
                 population_size: int = 50,
                 mutation_prob: float = 0.01,
                 elitism_count: int = 2,
                 max_generations: int = 100):
        """
        Initializes the Biogeography-Based Optimizer.
        """
        if not isinstance(hyperparameter_space, dict) or not hyperparameter_space:
            raise ValueError("hyperparameter_space must be a non-empty dictionary.")

        self.param_names: List[str] = list(hyperparameter_space.keys())
        self.param_configs: List[Dict] = [hyperparameter_space[name] for name in self.param_names]
        self.dimensions = len(self.param_names)

        super().__init__(name="BiogeographyBasedOptimizer", hyperparameter_count=self.dimensions)

        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.elitism_count = elitism_count
        self.max_generations = max_generations

        # Extract bounds for each parameter
        self.bounds = np.array([
            (c['min'], c['max']) if c['type'] in ['int', 'float']
            else (0, len(c['values']) - 1)
            for c in self.param_configs
        ]).T
        self.min_bounds, self.max_bounds = self.bounds[0], self.bounds[1]

        # BBO state
        self.population: np.ndarray | None = None
        self.fitness: np.ndarray | None = None
        self.current_generation_num = 0
        self.eval_idx = 0

        self._initialize_population()

    def _initialize_population(self):
        """Initializes the population of habitats."""
        self.population = np.random.uniform(
            low=self.min_bounds, high=self.max_bounds, size=(self.population_size, self.dimensions)
        )
        self.fitness = np.full(self.population_size, -np.inf)
        self.eval_idx = 0
        self.current_generation_num = 0
        print("Initialized BBO population (habitats).")

    def _decode_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        """Decodes a real-valued habitat vector into a dictionary of hyperparameters."""
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
        """Creates the next generation of habitats through migration and mutation."""
        print(f"--- Evolving to Generation {self.current_generation_num + 1} ---")

        # 1. Calculate Immigration and Emigration Rates
        sorted_indices = np.argsort(self.fitness)[::-1]
        sorted_fitness = self.fitness[sorted_indices]

        # Linear migration model
        max_rate = 1.0
        emigration_rates = max_rate * (1 - np.arange(self.population_size) / self.population_size)
        immigration_rates = max_rate * (np.arange(self.population_size) / self.population_size)

        # Map rates back to original population indices
        emigration_map = {idx: rate for idx, rate in zip(sorted_indices, emigration_rates)}
        immigration_map = {idx: rate for idx, rate in zip(sorted_indices, immigration_rates)}

        new_population = np.zeros_like(self.population)

        # 2. Elitism: Carry over the best habitats
        elite_indices = sorted_indices[:self.elitism_count]
        new_population[:self.elitism_count] = self.population[elite_indices]

        # 3. Migration for the rest of the population
        for i in range(self.elitism_count, self.population_size):
            target_habitat = np.copy(self.population[i])

            # Decide which features to migrate based on immigration rate
            for j in range(self.dimensions):
                if random.random() < immigration_map[i]:
                    # Select a source habitat based on emigration rates (roulette wheel)
                    emigration_probs = np.array([emigration_map[k] for k in range(self.population_size)])
                    emigration_probs /= np.sum(emigration_probs)

                    source_idx = np.random.choice(self.population_size, p=emigration_probs)

                    # Perform migration
                    target_habitat[j] = self.population[source_idx, j]

            new_population[i] = target_habitat

        # 4. Mutation on the non-elite part of the new population
        for i in range(self.elitism_count, self.population_size):
            for j in range(self.dimensions):
                if random.random() < self.mutation_prob:
                    new_population[i, j] = np.random.uniform(self.min_bounds[j], self.max_bounds[j])

        self.population = new_population
        self.fitness = np.full(self.population_size, -np.inf)
        self.eval_idx = 0
        self.current_generation_num += 1

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Suggests the next habitat to evaluate."""
        if self.eval_idx >= self.population_size:
            if self.current_generation_num < self.max_generations - 1:
                self._evolve()
            else:
                print("Max generations reached. Returning best found parameters.")
                return self.get_best_params() if self.best_params else {}

        habitat_vector = self.population[self.eval_idx]
        print(f"Gen {self.current_generation_num}, Suggesting habitat {self.eval_idx + 1}/{self.population_size}")
        return self._decode_vector(habitat_vector)

    def update(self, hyperparameters: Dict, score: float) -> None:
        """Updates the optimizer with the HSI (score) of the evaluated habitat."""
        self.fitness[self.eval_idx] = score

        self.history.append({'params': hyperparameters, 'score': score})
        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters

        print(f"Gen {self.current_generation_num}, Updated habitat {self.eval_idx + 1}/{self.population_size} "
              f"with score: {score:.4f}. Overall best: {self.get_best_score():.4f}")

        self.eval_idx += 1