from sklearn.ensemble import RandomForestClassifier
from typing import Dict
from models.BaseModel import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest model implementation"""

    def __init__(self):
        super().__init__("RandomForest")

    def define_hyperparameter_space(self) -> Dict[str, Dict]:
        return {
            "n_estimators": {"type": "int", "min": 1, "max": 500, "default": 100},
            "max_depth": {"type": "int", "min": 1, "max": 50, "default": 10},
            "min_samples_split": {"type": "int", "min": 2, "max": 100, "default": 2},
            "min_samples_leaf": {"type": "int", "min": 1, "max": 50, "default": 1},
            "max_features": {"type": "float", "min": 0.01, "max": 1.0, "default": 0.5},
        }

    def create_model(self, hyperparameters: Dict):
        self.model = RandomForestClassifier(
            n_estimators=int(hyperparameters["n_estimators"]),
            max_depth=int(hyperparameters["max_depth"]),
            min_samples_split=int(hyperparameters["min_samples_split"]),
            min_samples_leaf=int(hyperparameters["min_samples_leaf"]),
            max_features=hyperparameters["max_features"],
            random_state=42,
            n_jobs=-1,
        )
        return self.model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
