import random
from typing import Dict, List, Any
from optimizers.BaseOptimizer import BaseOptimizer


class GeneticAlgorithmOptimizer(BaseOptimizer):
    def __init__(self,
                 hyperparameter_space: Dict[str, Dict],
                 population_size: int = 50,
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.7,
                 generations: int = 100,
                 elitism_count: int = 1):
        """
        Initializes the Genetic Algorithm Optimizer.
        """
        if not isinstance(hyperparameter_space, dict) or not hyperparameter_space:
            raise ValueError("hyperparameter_space must be a non-empty dictionary.")

        # The GA internally uses an ordered list of parameter configurations.
        # We convert the input dict to ordered lists to ensure consistent mapping.
        self.param_names: List[str] = list(hyperparameter_space.keys())
        self.hyperparameter_space: List[Dict] = [hyperparameter_space[name] for name in self.param_names]

        chromosome_length = len(self.hyperparameter_space)

        if population_size <= 0:
            raise ValueError("Population size must be positive.")
        if chromosome_length <= 0:
            raise ValueError("Chromosome length derived from hyperparameter_space must be positive.")
        if not (0 <= mutation_rate <= 1):
            raise ValueError("Mutation rate must be between 0 and 1.")
        if not (0 <= crossover_rate <= 1):
            raise ValueError("Crossover rate must be between 0 and 1.")
        if generations <= 0:
            raise ValueError("Number of generations must be positive.")
        if elitism_count < 0 or elitism_count > population_size:
            raise ValueError("Elitism count must be non-negative and not exceed population size.")

        super().__init__(name="GeneticAlgorithmOptimizer", hyperparameter_count=chromosome_length)

        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = generations
        self.elitism_count = elitism_count

        self.population: List[List[Any]] = []
        self.current_generation_fitness: List[float] = []
        self.current_generation_num: int = 0

        self.last_suggested_chromosome: List[Any] | None = None
        self.best_chromosome: List[Any] | None = None

        self._initialize_population()

    def _decode_chromosome(self, chromosome: List[int]) -> Dict[str, Any]:
        """
        Decodes a binary chromosome into a dictionary of hyperparameters.
        This is a simplistic decoding strategy where 0 maps to the first choice (e.g., min)
        and 1 maps to the second choice (e.g., max).
        """
        params = {}
        if not chromosome or len(chromosome) != len(self.hyperparameter_space):
            print("Warning: Cannot decode chromosome due to length mismatch or it being empty.")
            return {}

        for i, gene in enumerate(chromosome):
            param_config = self.hyperparameter_space[i]
            param_name = self.param_names[i]  # Use the stored ordered list of names
            param_type = param_config.get('type')

            if param_type == 'int':
                value = param_config.get('min', 0) if gene == 0 else param_config.get('max', 1)
            elif param_type == 'float':
                value = param_config.get('min', 0.0) if gene == 0 else param_config.get('max', 1.0)
            elif param_type == 'categorical':
                values = param_config.get('values', [True, False])
                value = values[0] if gene == 0 else values[min(1, len(values) - 1)]
            else:
                value = gene

            params[param_name] = value
        return params

    def _initialize_population(self):
        """Initializes the population with random individuals and resets generation state."""
        self.population = []
        for _ in range(self.population_size):
            # Assuming binary chromosomes for simplicity; adapt if using other representations
            individual = [random.randint(0, 1) for _ in range(self.chromosome_length)]
            self.population.append(individual)
        self.current_generation_fitness = []  # Reset fitness list for the new population
        self.current_generation_num = 0  # Start at generation 0
        print(
            f"Initialized population for Generation {self.current_generation_num}. Population size: {self.population_size}")

    def _select_parents(self) -> tuple[List[Any] | None, List[Any] | None]:
        """Selects two parents from the population using tournament selection."""
        # This method assumes self.current_generation_fitness is populated for the current self.population
        if not self.population or not self.current_generation_fitness or len(self.population) != len(
                self.current_generation_fitness):
            print("Warning: Parent selection called with inconsistent population/fitness data.")
            return (random.choice(self.population) if self.population else None,
                    random.choice(self.population) if self.population else None)

        tournament_size = min(5, len(self.population))  # Ensure tournament size isn't larger than population
        if tournament_size == 0:
            return None, None

        parents = []
        for _ in range(2):  # Select two parents
            tournament_contestants_indices = random.sample(range(len(self.population)), tournament_size)
            tournament_winner_index = -1
            best_tournament_fitness = -float('inf')

            for index in tournament_contestants_indices:
                if self.current_generation_fitness[index] > best_tournament_fitness:
                    best_tournament_fitness = self.current_generation_fitness[index]
                    tournament_winner_index = index

            if tournament_winner_index != -1:
                parents.append(self.population[tournament_winner_index])
            else:  # Fallback if all contestants had -inf fitness or other issue
                parents.append(random.choice(self.population))
        return parents[0], parents[1]

    def _crossover(self, parent1: List[Any], parent2: List[Any]) -> tuple[List[Any], List[Any]]:
        """Performs single-point crossover between two parents."""
        if random.random() < self.crossover_rate and self.chromosome_length > 1:
            point = random.randint(1, self.chromosome_length - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return list(parent1), list(parent2)  # Return copies if no crossover

    def _mutate(self, individual: List[Any]) -> List[Any]:
        """Performs bit-flip mutation on an individual."""
        mutated_individual = list(individual)  # Create a mutable copy
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                # Assuming binary chromosome; adapt for other types
                mutated_individual[i] = 1 - mutated_individual[i]
        return mutated_individual

    def _evolve_new_population(self):
        """Generates a new population through elitism, selection, crossover, and mutation."""
        new_population = []

        # Elitism: Carry over the best individuals
        if self.elitism_count > 0 and self.current_generation_fitness:
            # Ensure population and fitness lists are aligned
            if len(self.population) == len(self.current_generation_fitness):
                sorted_population_indices = sorted(range(len(self.population)),
                                                   key=lambda k: self.current_generation_fitness[k],
                                                   reverse=True)
                for i in range(min(self.elitism_count, len(self.population))):
                    elite_idx = sorted_population_indices[i]
                    new_population.append(list(self.population[elite_idx]))  # Store copy
            else:
                print("Warning: Elitism skipped due to mismatch in population and fitness data lengths.")

        # Fill the rest of the population
        while len(new_population) < self.population_size:
            parent1, parent2 = self._select_parents()
            if parent1 is None or parent2 is None:  # Should only happen if population is tiny/empty
                print("Warning: Parent selection failed, adding random individuals.")
                if self.population:
                    parent1 = random.choice(self.population)
                    parent2 = random.choice(self.population)
                else:  # Critical: population became empty
                    self._initialize_population()  # Attempt recovery
                    if not self.population: break  # Still empty, cannot proceed
                    parent1, parent2 = random.choice(self.population), random.choice(self.population)

            child1, child2 = self._crossover(parent1, parent2)
            mutated_child1 = self._mutate(child1)
            new_population.append(mutated_child1)

            if len(new_population) < self.population_size:
                mutated_child2 = self._mutate(child2)
                new_population.append(mutated_child2)

        self.population = new_population[:self.population_size]
        self.current_generation_fitness = []  # Clear fitness for the new generation
        self.current_generation_num += 1
        print(f"--- Evolved to Generation {self.current_generation_num} ---")

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """
        Suggests the next set of hyperparameters by decoding a chromosome.
        """
        # 1. Evolve population if a generation is complete
        if len(self.current_generation_fitness) == self.population_size:
            if self.current_generation_num < self.max_generations - 1:
                self._evolve_new_population()
            else:
                # Max generations reached, return the best found
                print("Max generations reached. Suggesting best found individual.")
                if self.best_chromosome:
                    return self._decode_chromosome(self.best_chromosome)
                # Fallback if no best chromosome was ever recorded
                return self._decode_chromosome(random.choice(self.population)) if self.population else {}

        # 2. Suggest the next individual from the current population
        idx_to_suggest = len(self.current_generation_fitness)

        if not self.population or idx_to_suggest >= len(self.population):
            # Handle unexpected empty population
            self._initialize_population()
            idx_to_suggest = 0

        individual = self.population[idx_to_suggest]
        self.last_suggested_chromosome = individual  # Track for the update step

        print(f"Gen {self.current_generation_num}, Suggesting individual {idx_to_suggest + 1}/{self.population_size}")
        return self._decode_chromosome(individual)

    def get_best_individual(self) -> List[Any] | None:
        """Returns the best individual chromosome found."""
        best_params_dict = self.get_best_params()
        if best_params_dict and 'chromosome' in best_params_dict:
            return best_params_dict['chromosome']
        return None

    def update(self, hyperparameters: Dict, score: float) -> None:
        """
        Updates the optimizer with the fitness score of a suggested chromosome.
        """
        if self.last_suggested_chromosome is None:
            print("Warning: Update called without a recently suggested chromosome. Skipping.")
            return

        # Update GA-specific state
        self.current_generation_fitness.append(score)

        # Update BaseOptimizer state
        self.history.append({'params': hyperparameters, 'score': score})
        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters
            self.best_chromosome = self.last_suggested_chromosome  # Save the best chromosome

        print(
            f"Gen {self.current_generation_num}, Updated individual {len(self.current_generation_fitness)}/{self.population_size} "
            f"with score: {score:.4f}. Overall best: {self.get_best_score():.4f}")