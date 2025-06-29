import numpy as np
from typing import Dict, List
from optimizers.BaseOptimizer import BaseOptimizer


class GridSearch(BaseOptimizer):
    """Grid search optimization strategy"""

    def __init__(self, hyperparameter_space: Dict):
        super().__init__("GridSearch", len(hyperparameter_space))
        self.hyperparameter_space = hyperparameter_space
        self.grid_points = self._generate_grid()
        self.current_index = 0

    def _generate_grid(self) -> List[Dict]:
        """Generate all grid points"""
        # Simplified grid generation - create 3-5 points per parameter
        grid = []
        param_grids = {}

        for param_name, param_config in self.hyperparameter_space.items():
            if param_config["type"] == "int":
                param_grids[param_name] = np.linspace(
                    param_config["min"], param_config["max"], 3, dtype=int
                ).tolist()
            elif param_config["type"] == "float":
                param_grids[param_name] = np.linspace(
                    param_config["min"], param_config["max"], 3
                ).tolist()
            elif param_config["type"] == "categorical":
                param_grids[param_name] = param_config["values"]

        # Generate all combinations (limited to prevent explosion)
        import itertools

        param_names = list(param_grids.keys())
        param_values = [param_grids[name] for name in param_names]

        for combination in itertools.product(*param_values):
            grid.append(dict(zip(param_names, combination)))

        return grid[:50]  # Limit to first 50 combinations

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Return next grid point"""
        if self.current_index >= len(self.grid_points):
            # If we've exhausted the grid, start over
            self.current_index = 0

        suggestion = self.grid_points[self.current_index]
        self.current_index += 1
        return suggestion

    def update(self, hyperparameters: Dict, score: float):
        """Update with new result"""
        self.history.append({"params": hyperparameters, "score": score})

        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters.copy()
