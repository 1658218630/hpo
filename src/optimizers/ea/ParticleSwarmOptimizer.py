import numpy as np
import random
from typing import Dict, List, Any
from optimizers.BaseOptimizer import BaseOptimizer


class ParticleSwarmOptimizer(BaseOptimizer):
    """
    An optimizer based on the Particle Swarm Optimization (PSO) algorithm.
    """

    def __init__(self,
                 hyperparameter_space: Dict[str, Dict],
                 swarm_size: int = 30,
                 inertia_weight: float = 0.5,  # w: momentum
                 cognitive_coeff: float = 1.5, # c1: pull towards personal best
                 social_coeff: float = 1.5,    # c2: pull towards global best
                 max_generations: int = 100):
        """
        Initializes the Particle Swarm Optimizer.
        """
        if not isinstance(hyperparameter_space, dict) or not hyperparameter_space:
            raise ValueError("hyperparameter_space must be a non-empty dictionary.")

        self.param_names: List[str] = list(hyperparameter_space.keys())
        self.param_configs: List[Dict] = [hyperparameter_space[name] for name in self.param_names]
        self.dimensions = len(self.param_names)

        super().__init__(name="ParticleSwarmOptimizer", hyperparameter_count=self.dimensions)

        self.swarm_size = swarm_size
        self.w = inertia_weight
        self.c1 = cognitive_coeff
        self.c2 = social_coeff
        self.max_generations = max_generations

        # Extract bounds for each parameter
        self.bounds = np.array([
            (c['min'], c['max']) if c['type'] in ['int', 'float']
            else (0, len(c['values']) - 1)
            for c in self.param_configs
        ]).T
        self.min_bounds, self.max_bounds = self.bounds[0], self.bounds[1]

        # PSO-specific state
        self.positions: np.ndarray | None = None
        self.velocities: np.ndarray | None = None
        self.pbest_positions: np.ndarray | None = None
        self.pbest_scores: np.ndarray | None = None
        self.gbest_position: np.ndarray | None = None
        self.gbest_score: float = -np.inf

        self.current_generation_num = 0
        self.eval_idx = 0  # Index of the particle to evaluate next

        self._initialize_swarm()

    def _initialize_swarm(self):
        """Initializes the swarm's positions, velocities, and bests."""
        # Initialize positions randomly within bounds
        self.positions = np.random.uniform(
            low=self.min_bounds, high=self.max_bounds, size=(self.swarm_size, self.dimensions)
        )
        # Initialize velocities to a small random value
        vel_range = self.max_bounds - self.min_bounds
        self.velocities = np.random.uniform(
            low=-vel_range, high=vel_range, size=(self.swarm_size, self.dimensions)
        ) * 0.1

        # Initialize personal bests
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.full(self.swarm_size, -np.inf)

        # Initialize global best (will be updated after first evaluations)
        self.gbest_position = self.pbest_positions[0].copy()
        self.gbest_score = -np.inf

        self.eval_idx = 0
        self.current_generation_num = 0
        print("Initialized Particle Swarm.")

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

    def _update_swarm_state(self):
        """Updates the velocity and position of all particles in the swarm."""
        print(f"--- Updating Swarm for Generation {self.current_generation_num + 1} ---")

        r1 = np.random.rand(self.swarm_size, self.dimensions)
        r2 = np.random.rand(self.swarm_size, self.dimensions)

        # Update velocities
        cognitive_velocity = self.c1 * r1 * (self.pbest_positions - self.positions)
        social_velocity = self.c2 * r2 * (self.gbest_position - self.positions)
        self.velocities = self.w * self.velocities + cognitive_velocity + social_velocity

        # Update positions
        self.positions += self.velocities

        # Clamp positions to stay within bounds
        self.positions = np.clip(self.positions, self.min_bounds, self.max_bounds)

        self.eval_idx = 0
        self.current_generation_num += 1

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Suggests the next particle's position to evaluate."""
        if self.eval_idx >= self.swarm_size:
            if self.current_generation_num < self.max_generations - 1:
                self._update_swarm_state()
            else:
                print("Max generations reached. Returning best found parameters.")
                return self.get_best_params() if self.best_params else {}

        particle_position = self.positions[self.eval_idx]
        print(f"Gen {self.current_generation_num}, Suggesting particle {self.eval_idx + 1}/{self.swarm_size}")
        return self._decode_vector(particle_position)

    def update(self, hyperparameters: Dict, score: float) -> None:
        """Updates the swarm with the fitness score of the evaluated particle."""
        # Update personal best for the current particle
        if score > self.pbest_scores[self.eval_idx]:
            self.pbest_scores[self.eval_idx] = score
            self.pbest_positions[self.eval_idx] = self.positions[self.eval_idx]

        # Update global best
        if score > self.gbest_score:
            self.gbest_score = score
            self.gbest_position = self.positions[self.eval_idx]
            # Also update the base class bests
            self.best_score = score
            self.best_params = hyperparameters

        self.history.append({'params': hyperparameters, 'score': score})

        print(f"Gen {self.current_generation_num}, Updated particle {self.eval_idx + 1}/{self.swarm_size} "
              f"with score: {score:.4f}. Swarm best: {self.gbest_score:.4f}")

        self.eval_idx += 1