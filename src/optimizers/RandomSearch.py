import numpy as np
from optimizers.BaseOptimizer import BaseOptimizer
from typing import Dict


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, hyperparameter_space: Dict):
        super().__init__("RandomSearch", len(hyperparameter_space))
        self.hyperparameter_space = hyperparameter_space

    def suggest_hyperparameters(self, round_num: int) -> Dict:
        suggestion = {}

        for param_name, param_config in self.hyperparameter_space.items():
            if param_config["type"] == "int":
                suggestion[param_name] = np.random.randint(
                    param_config["min"], param_config["max"] + 1
                )
            elif param_config["type"] == "float":
                suggestion[param_name] = np.random.uniform(
                    param_config["min"], param_config["max"]
                )
            elif param_config["type"] == "categorical":
                # Fix: Use random.choice() instead of np.random.choice() for complex objects
                import random

                suggestion[param_name] = random.choice(param_config["values"])

        return suggestion

    def update(self, hyperparameters: Dict, score: float):
        self.history.append({"params": hyperparameters, "score": score})

        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparameters.copy()
