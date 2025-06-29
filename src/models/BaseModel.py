from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseModel(ABC):
    """Abstract base class for all machine learning models"""

    def __init__(self, name: str):
        self.name = name
        self.hyperparameter_space = self.define_hyperparameter_space()
        self.model = None

    @abstractmethod
    def define_hyperparameter_space(self) -> Dict[str, Dict]:
        """Define the hyperparameter search space"""
        pass

    @abstractmethod
    def create_model(self, hyperparameters: Dict) -> Any:
        """Create model instance with given hyperparameters"""
        pass

    @abstractmethod
    def fit(self, X_train, y_train) -> None:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Make predictions"""
        pass

    def get_hyperparameter_count(self) -> int:
        """Return total number of hyperparameters"""
        return len(self.hyperparameter_space)

    def get_hyperparameter_names(self) -> List[str]:
        """Return list of hyperparameter names"""
        return list(self.hyperparameter_space.keys())
