from catboost import CatBoostClassifier
from typing import Dict
from models.BaseModel import BaseModel


class CatBoostModel(BaseModel):
    """
    CatBoost model - robust gradient boosting
    Why CatBoost is excellent for HPO:
    - Very robust to hyperparameters (harder to break)
    - Built-in categorical feature handling
    - Strong default performance
    - Good hyperparameter interactions for RL to learn
    """

    def __init__(self):
        super().__init__("CatBoost")

    def define_hyperparameter_space(self) -> Dict[str, Dict]:
        return {
            # Core parameters
            "iterations": {
                "type": "int",
                "min": 50,
                "max": 500,
                "default": 100,
            },
            "depth": {
                "type": "int",
                "min": 4,
                "max": 10,
                "default": 6,
            },
            "learning_rate": {
                "type": "float",
                "min": 0.01,
                "max": 0.3,
                "default": 0.1,
            },
            # Regularization
            "l2_leaf_reg": {
                "type": "float",
                "min": 1.0,
                "max": 30.0,
                "default": 3.0,
            },
            "bootstrap_type": {
                "type": "categorical",
                "values": ["Bayesian", "Bernoulli", "MVS"],
                "default": "MVS",
            },
            # Sampling parameters - IMMER verfügbar, aber nur relevant bei bestimmten bootstrap_types
            "subsample": {
                "type": "float",
                "min": 0.5,
                "max": 1.0,
                "default": 1.0,
            },
            "rsm": {  # Random subspace method
                "type": "float",
                "min": 0.5,
                "max": 1.0,
                "default": 1.0,
            },
            # Tree parameters
            "min_data_in_leaf": {
                "type": "int",
                "min": 1,
                "max": 50,
                "default": 1,
            },
        }

    def create_model(self, hyperparameters: Dict):
        # Vereinfachte Parameter-Behandlung
        model_params = {
            "iterations": int(hyperparameters["iterations"]),
            "depth": int(hyperparameters["depth"]),
            "learning_rate": hyperparameters["learning_rate"],
            "l2_leaf_reg": hyperparameters["l2_leaf_reg"],
            "bootstrap_type": hyperparameters["bootstrap_type"],
            "rsm": hyperparameters["rsm"],
            "min_data_in_leaf": int(hyperparameters["min_data_in_leaf"]),
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,  # Prevents CatBoost from creating files
        }

        # Nur subsample hinzufügen wenn es vom bootstrap_type verwendet wird
        if hyperparameters["bootstrap_type"] == "Bernoulli":
            model_params["subsample"] = hyperparameters["subsample"]

        self.model = CatBoostClassifier(**model_params)
        return self.model

    def fit(self, X_train, y_train):
        # VEREINFACHT: Kein manuelles Early Stopping in Cross-Validation
        # CatBoost macht automatisch overfitting protection
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
