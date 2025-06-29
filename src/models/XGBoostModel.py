import xgboost as xgb
from typing import Dict
from models.BaseModel import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model - excellent for hyperparameter optimization"""

    def __init__(self):
        super().__init__("XGBoost")

    def define_hyperparameter_space(self) -> Dict[str, Dict]:
        return {
            "max_depth": {"type": "int", "min": 3, "max": 10, "default": 6},
            "n_estimators": {"type": "int", "min": 50, "max": 500, "default": 100},
            "learning_rate": {"type": "float", "min": 0.01, "max": 0.3, "default": 0.1},
            "subsample": {"type": "float", "min": 0.6, "max": 1.0, "default": 1.0},
            "colsample_bytree": {
                "type": "float",
                "min": 0.6,
                "max": 1.0,
                "default": 1.0,
            },
            "reg_alpha": {"type": "float", "min": 0.0, "max": 10.0, "default": 0.0},
            "reg_lambda": {"type": "float", "min": 0.0, "max": 10.0, "default": 1.0},
            "min_child_weight": {"type": "int", "min": 1, "max": 10, "default": 1},
        }

    def create_model(self, hyperparameters: Dict):
        self.model = xgb.XGBClassifier(
            max_depth=int(hyperparameters["max_depth"]),
            n_estimators=int(hyperparameters["n_estimators"]),
            learning_rate=hyperparameters["learning_rate"],
            subsample=hyperparameters["subsample"],
            colsample_bytree=hyperparameters["colsample_bytree"],
            reg_alpha=hyperparameters["reg_alpha"],
            reg_lambda=hyperparameters["reg_lambda"],
            min_child_weight=int(hyperparameters["min_child_weight"]),
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )
        return self.model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
