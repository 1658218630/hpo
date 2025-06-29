import lightgbm as lgb
from typing import Dict
from models.BaseModel import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM model - fastest gradient boosting
    Why LightGBM is great for HPO:
    - Extremely fast training (2-10x faster than XGBoost)
    - Many hyperparameters with complex interactions
    - Built-in early stopping
    - Excellent performance on tabular data
    """

    def __init__(self):
        super().__init__("LightGBM")

    def define_hyperparameter_space(self) -> Dict[str, Dict]:
        return {
            # Core tree parameters
            "num_leaves": {
                "type": "int",
                "min": 10,
                "max": 300,
                "default": 31,
            },
            "max_depth": {
                "type": "int",
                "min": 3,
                "max": 15,
                "default": -1,  # -1 means no limit
            },
            "n_estimators": {
                "type": "int",
                "min": 50,
                "max": 500,
                "default": 100,
            },
            # Learning parameters
            "learning_rate": {
                "type": "float",
                "min": 0.01,
                "max": 0.3,
                "default": 0.1,
            },
            "feature_fraction": {
                "type": "float",
                "min": 0.5,
                "max": 1.0,
                "default": 1.0,
            },
            "bagging_fraction": {
                "type": "float",
                "min": 0.5,
                "max": 1.0,
                "default": 1.0,
            },
            "bagging_freq": {
                "type": "int",
                "min": 0,
                "max": 7,
                "default": 0,
            },
            # Regularization
            "lambda_l1": {
                "type": "float",
                "min": 0.0,
                "max": 10.0,
                "default": 0.0,
            },
            "lambda_l2": {
                "type": "float",
                "min": 0.0,
                "max": 10.0,
                "default": 0.0,
            },
            "min_child_samples": {
                "type": "int",
                "min": 5,
                "max": 100,
                "default": 20,
            },
        }

    def create_model(self, hyperparameters: Dict):
        # Conditional parameter handling für bagging
        model_params = {
            "num_leaves": int(hyperparameters["num_leaves"]),
            "max_depth": int(hyperparameters["max_depth"]),
            "n_estimators": int(hyperparameters["n_estimators"]),
            "learning_rate": hyperparameters["learning_rate"],
            "feature_fraction": hyperparameters["feature_fraction"],
            "lambda_l1": hyperparameters["lambda_l1"],
            "lambda_l2": hyperparameters["lambda_l2"],
            "min_child_samples": int(hyperparameters["min_child_samples"]),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
            "force_col_wise": True,  # Prevents threading warnings
        }

        # Bagging parameters nur hinzufügen wenn bagging_freq > 0
        if hyperparameters["bagging_freq"] > 0:
            model_params["bagging_fraction"] = hyperparameters["bagging_fraction"]
            model_params["bagging_freq"] = int(hyperparameters["bagging_freq"])

        self.model = lgb.LGBMClassifier(**model_params)
        return self.model

    def fit(self, X_train, y_train):
        # VEREINFACHT: Kein manuelles Early Stopping in Cross-Validation
        # Das war problematisch bei Cross-Validation wie bei CatBoost
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
