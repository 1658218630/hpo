from sklearn.svm import SVC
from typing import Dict
from models.BaseModel import BaseModel


class SVMModel(BaseModel):
    """Support Vector Machine model implementation"""

    def __init__(self):
        super().__init__("SVM")

    def define_hyperparameter_space(self) -> Dict[str, Dict]:
        return {
            "C": {"type": "float", "min": 0.01, "max": 100.0, "default": 1.0},
            "gamma": {"type": "float", "min": 0.001, "max": 10.0, "default": 1.0},
            "kernel": {
                "type": "categorical",
                "values": ["rbf", "linear", "poly"],
                "default": "rbf",
            },
        }

    def create_model(self, hyperparameters: Dict):
        self.model = SVC(
            C=hyperparameters["C"],
            gamma=hyperparameters["gamma"],
            kernel=hyperparameters["kernel"],
            random_state=42,
        )
        return self.model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
