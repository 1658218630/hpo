# src/optimizers/ea/NEATOptimizer.py

import random
import numpy as np
from typing import Dict, List, Any
from optimizers.BaseOptimizer import BaseOptimizer


# --- NEAT-inspired Components for Hyperparameter Tuning ---

class HyperparameterGene:
    """Represents a single hyperparameter within a genome."""

    def __init__(self, name: str, value: Any, config: Dict):
        self.name = name
        self.value = value
        self.config = config  # Stores type, range, etc.

    def copy(self):
        """Creates a deep copy of the gene."""
        return HyperparameterGene(self.name, self.value, self.config)

    def mutate_value(self, mutation_strength: float = 0.1):
        """Mutates the gene's value according to its type."""
        config_type = self.config['type']
        if config_type == 'float':
            noise = np.random.normal(0, (self.config['max'] - self.config['min']) * mutation_strength)
            self.value = np.clip(self.value + noise, self.config['min'], self.config['max'])
        elif config_type == 'int':
            # Use geometric distribution to favor smaller steps
            step = int(np.random.geometric(p=0.5)) * random.choice([-1, 1])
            self.value = int(np.clip(self.value + step, self.config['min'], self.config['max']))
        elif config_type == 'categorical':
            self.value = random.choice(self.config['values'])


class Genome:
    """Represents a candidate solution (a set of hyperparameter genes)."""

    def __init__(self, all_params_config: Dict[str, Dict]):
        self.genes: Dict[str, HyperparameterGene] = {}  # name -> gene
        self.all_params_config = all_params_config
        self.fitness = -np.inf

    def copy(self):
        """Creates a deep copy of the genome."""
        new_genome = Genome(self.all_params_config)
        new_genome.genes = {name: gene.copy() for name, gene in self.genes.items()}
        new_genome.fitness = self.fitness
        return new_genome

    def to_dict(self) -> Dict[str, Any]:
        """Decodes the genome into a hyperparameter dictionary for model evaluation."""
        return {name: gene.value for name, gene in self.genes.items()}


class NEATOptimizer(BaseOptimizer):
    """
    A NEAT-inspired optimizer adapted for hyperparameter tuning.

    This optimizer evolves a population of genomes, where each genome represents
    a subset of the total available hyperparameters. It uses mutation to not only
    tweak hyperparameter values but also to add new hyperparameters to a solution,
    thus "augmenting the topology" of the hyperparameter set.
    """

    def __init__(self,
                 hyperparameter_space: Dict[str, Dict],
                 population_size: int = 50,
                 max_generations: int = 100,
                 elitism_count: int = 2,
                 add_gene_prob: float = 0.1,
                 mutate_value_prob: float = 0.8):
        """Initializes the NEAT-inspired Optimizer."""
        if not isinstance(hyperparameter_space, dict) or not hyperparameter_space:
            raise ValueError("hyperparameter_space must be a non-empty dictionary.")

        super().__init__(name="NEATOptimizer", hyperparameter_count=len(hyperparameter_space))

        self.all_params_config = {name: config for name, config in hyperparameter_space.items()}
        self.param_names = list(self.all_params_config.keys())

        self.population_size = population_size
        self.max_generations = max_generations
        self.elitism_count = elitism_count
        self.add_gene_prob = add_gene_prob
        self.mutate_value_prob = mutate_value_prob

        self.default_params = self._create_default_params()
        self.population: List[Genome] = self._initialize_population()

        self.population: List[Genome] = self._initialize_population()
        self.fitness_scores = np.full(self.population_size, -np.inf)
        self.current_generation_num = 0
        self.eval_idx = 0

    def _create_default_params(self) -> Dict[str, Any]:
        """Creates a dictionary of default hyperparameters from the space."""
        defaults = {}
        for name, config in self.all_params_config.items():
            if config['type'] == 'float':
                # Use the middle of the range or a common default like 0.1
                defaults[name] = (config['min'] + config['max']) / 2
            elif config['type'] == 'int':
                # Use the middle of the range
                defaults[name] = int((config['min'] + config['max']) / 2)
            elif config['type'] == 'categorical':
                # Use the first value in the list
                defaults[name] = config['values'][0]
        return defaults

    def _initialize_population(self) -> List[Genome]:
        """Creates the initial population of minimal genomes."""
        pop = []
        for _ in range(self.population_size):
            genome = Genome(self.all_params_config)
            # Start each genome with 1 to 3 random hyperparameters
            num_initial_genes = random.randint(1, min(3, len(self.param_names)))
            initial_gene_names = random.sample(self.param_names, num_initial_genes)
            for name in initial_gene_names:
                self._add_gene_to_genome(genome, name)
            pop.append(genome)
        print(f"Initialized NEAT population with {self.population_size} genomes.")
        return pop

    def _add_gene_to_genome(self, genome: Genome, param_name: str):
        """Adds a new gene with a random value to a genome."""
        config = self.all_params_config[param_name]
        val = None
        if config['type'] == 'float':
            val = random.uniform(config['min'], config['max'])
        elif config['type'] == 'int':
            val = random.randint(config['min'], config['max'])
        elif config['type'] == 'categorical':
            val = random.choice(config['values'])

        if val is not None:
            genome.genes[param_name] = HyperparameterGene(param_name, val, config)

    def _mutate(self, genome: Genome):
        """Applies mutation to a genome: either adds a new gene or mutates an existing one."""
        # Chance to add a new gene (augment topology)
        if random.random() < self.add_gene_prob and len(genome.genes) < len(self.param_names):
            possible_new_genes = set(self.param_names) - set(genome.genes.keys())
            if possible_new_genes:
                new_gene_name = random.choice(list(possible_new_genes))
                self._add_gene_to_genome(genome, new_gene_name)

        # Chance to mutate existing gene values
        if random.random() < self.mutate_value_prob and genome.genes:
            gene_to_mutate = random.choice(list(genome.genes.values()))
            gene_to_mutate.mutate_value()

    def _crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Performs crossover between two parent genomes."""
        if parent1.fitness < parent2.fitness:
            parent1, parent2 = parent2, parent1  # Ensure parent1 is the fitter one

        child = Genome(self.all_params_config)
        p1_genes = parent1.genes
        p2_genes = parent2.genes

        # Inherit all genes from the fitter parent
        for name, gene1 in p1_genes.items():
            child.genes[name] = gene1.copy()
            # If the less fit parent also has the gene, 50% chance to take its value
            if name in p2_genes and random.random() < 0.5:
                child.genes[name].value = p2_genes[name].value

        return child

    def _evolve(self):
        """Creates the next generation of genomes."""
        print(f"--- Evolving to Generation {self.current_generation_num + 1} ---")
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        new_population = []

        # Elitism: Carry over the best genomes
        for i in range(self.elitism_count):
            new_population.append(self.population[i].copy())

        # Create the rest of the new population
        while len(new_population) < self.population_size:
            # Select parents from the top half of the population
            parent1 = random.choice(self.population[:self.population_size // 2])
            parent2 = random.choice(self.population[:self.population_size // 2])
            child = self._crossover(parent1, parent2)
            self._mutate(child)
            new_population.append(child)

        self.population = new_population
        self.fitness_scores = np.full(self.population_size, -np.inf)
        self.eval_idx = 0
        self.current_generation_num += 1

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Suggests the next genome's hyperparameters to be evaluated."""
        if self.eval_idx >= self.population_size:
            if self.current_generation_num < self.max_generations - 1:
                self._evolve()
            else:
                print("Max generations reached. Returning best found parameters.")
                # Ensure even the best params are merged with defaults
                best_genome_params = self.get_best_params() if self.best_params else {}
                final_params = self.default_params.copy()
                final_params.update(best_genome_params)
                return final_params

        genome_to_eval = self.population[self.eval_idx]

        full_params = self.default_params.copy()
        genome_params = genome_to_eval.to_dict()

        # Overwrite the defaults with the evolved parameters
        full_params.update(genome_params)

        print(f"Gen {self.current_generation_num}, Suggesting genome {self.eval_idx + 1}/{self.population_size} "
              f"with {len(genome_to_eval.genes)} params")

        return full_params  # Return the complete dictionary

    def update(self, hyperparameters: Dict, score: float) -> None:
        """Updates the fitness of the evaluated genome."""
        if self.eval_idx >= self.population_size:
            return  # Avoid index out of bounds if update is called unexpectedly

        self.population[self.eval_idx].fitness = score
        self.fitness_scores[self.eval_idx] = score

        self.history.append({'params': hyperparameters, 'score': score})
        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters

        print(f"Gen {self.current_generation_num}, Updated genome {self.eval_idx + 1}/{self.population_size} "
              f"with score: {score:.4f}. Overall best: {self.get_best_score():.4f}")

        self.eval_idx += 1