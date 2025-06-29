import numpy as np
from abc import ABC, abstractmethod
from typing import Dict


class BaseOptimizer(ABC):
    """Abstract base class for all optimization strategies"""

    def __init__(self, name: str, hyperparameter_count: int):
        self.name = name
        self.hyperparameter_count = hyperparameter_count
        self.history = []
        self.best_score = -np.inf
        self.best_params = None

    @abstractmethod
    def suggest_hyperparameters(self, round_num: int) -> Dict:
        """Suggest next set of hyperparameters to try"""
        pass

    @abstractmethod
    def update(self, hyperparameters: Dict, score: float) -> None:
        """Update optimizer with results from evaluation"""
        pass

    def get_best_score(self) -> float:
        """Return best score achieved so far"""
        return self.best_score

    def get_best_params(self) -> Dict:
        """Return best hyperparameters found so far"""
        return self.best_params
