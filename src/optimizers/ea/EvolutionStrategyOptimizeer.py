import numpy as np
import random
from typing import Dict, List, Any
from optimizers.BaseOptimizer import BaseOptimizer


class EvolutionStrategyOptimizer(BaseOptimizer):
    """
    An optimizer based on a (μ+λ) Evolution Strategy (ES).

    In this strategy, μ parents generate λ offspring. The next generation of
    parents is selected from the best individuals of the combined population
    of both parents and offspring.
    """

    def __init__(self,
                 hyperparameter_space: Dict[str, Dict],
                 population_size: int = 10,  # Number of parents (μ)
                 offspring_size: int = 40,   # Number of offspring to generate (λ)
                 initial_sigma: float = 0.5, # Initial mutation strength
                 learning_rate: float = 0.95, # Rate to adapt sigma (closer to 1 is slower)
                 max_generations: int = 100):
        """
        Initializes the Evolution Strategy Optimizer.
        """
        if not isinstance(hyperparameter_space, dict) or not hyperparameter_space:
            raise ValueError("hyperparameter_space must be a non-empty dictionary.")

        self.param_names: List[str] = list(hyperparameter_space.keys())
        self.param_configs: List[Dict] = [hyperparameter_space[name] for name in self.param_names]
        self.chromosome_length = len(self.param_names)

        super().__init__(name="EvolutionStrategyOptimizer", hyperparameter_count=self.chromosome_length)

        self.population_size = population_size
        self.offspring_size = offspring_size
        self.sigma = initial_sigma
        self.learning_rate = learning_rate
        self.max_generations = max_generations

        # Extract bounds for each parameter for clipping
        self.bounds = []
        for config in self.param_configs:
            if config['type'] in ['int', 'float']:
                self.bounds.append((config['min'], config['max']))
            else:  # Categorical
                self.bounds.append((0, len(config['values']) - 1))
        self.min_bounds = np.array([b[0] for b in self.bounds])
        self.max_bounds = np.array([b[1] for b in self.bounds])

        self.population: np.ndarray | None = None
        self.population_fitness: np.ndarray | None = None
        self.offspring: np.ndarray | None = None
        self.offspring_fitness: List[float] = []

        self.current_generation_num = 0
        self.eval_idx = 0

        self._initialize_population()
        self._generate_offspring()

    def _initialize_population(self):
        """Initializes the parent population with random real-valued vectors."""
        self.population = np.zeros((self.population_size, self.chromosome_length))
        for i in range(self.population_size):
            for j in range(self.chromosome_length):
                min_b, max_b = self.bounds[j]
                self.population[i, j] = random.uniform(min_b, max_b)
        self.population_fitness = np.full(self.population_size, -np.inf)
        print("Initialized Evolution Strategy parent population.")

    def _decode_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        """Decodes a real-valued vector into a dictionary of hyperparameters."""
        params = {}
        for i, value in enumerate(vector):
            name = self.param_names[i]
            config = self.param_configs[i]
            min_b, max_b = self.bounds[i]

            value = np.clip(value, min_b, max_b)

            if config['type'] == 'int':
                params[name] = int(round(value))
            elif config['type'] == 'float':
                params[name] = float(value)
            elif config['type'] == 'categorical':
                idx = int(round(np.clip(value, min_b, max_b)))
                params[name] = config['values'][idx]
        return params

    def _generate_offspring(self):
        """Generates λ new offspring from the parent population via mutation."""
        self.offspring = np.zeros((self.offspring_size, self.chromosome_length))
        for i in range(self.offspring_size):
            parent = self.population[random.randint(0, self.population_size - 1)]
            noise = np.random.normal(0, self.sigma, self.chromosome_length)
            offspring_candidate = parent + noise
            self.offspring[i] = np.clip(offspring_candidate, self.min_bounds, self.max_bounds)

        self.offspring_fitness = []
        self.eval_idx = 0

    def _evolve(self):
        """Selects the next generation of parents and adapts sigma."""
        print(f"--- Evolving to Generation {self.current_generation_num + 1} ---")

        combined_population = np.vstack((self.population, self.offspring))
        combined_fitness = np.concatenate((self.population_fitness, np.array(self.offspring_fitness)))

        sorted_indices = np.argsort(combined_fitness)[::-1]

        new_parent_indices = sorted_indices[:self.population_size]
        self.population = combined_population[new_parent_indices]
        self.population_fitness = combined_fitness[new_parent_indices]

        # Adapt sigma using the 1/5 success rule
        avg_parent_fitness = np.mean(self.population_fitness) if len(self.population_fitness) > 0 else -np.inf
        successes = np.sum(np.array(self.offspring_fitness) > avg_parent_fitness)
        success_ratio = successes / self.offspring_size

        if success_ratio > 0.2:
            self.sigma /= self.learning_rate
            print(f"Success ratio {success_ratio:.2f} > 0.2, increasing sigma to {self.sigma:.4f}")
        else:
            self.sigma *= self.learning_rate
            print(f"Success ratio {success_ratio:.2f} <= 0.2, decreasing sigma to {self.sigma:.4f}")

        self._generate_offspring()
        self.current_generation_num += 1

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Suggests the next set of hyperparameters to evaluate."""
        if self.eval_idx >= self.offspring_size:
            if self.current_generation_num < self.max_generations - 1:
                self._evolve()
            else:
                print("Max generations reached. Returning best found parameters.")
                return self.get_best_params() if self.best_params else {}

        individual_vector = self.offspring[self.eval_idx]
        print(f"Gen {self.current_generation_num}, Suggesting offspring {self.eval_idx + 1}/{self.offspring_size}")
        return self._decode_vector(individual_vector)

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