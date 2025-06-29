from sklearn.neural_network import MLPClassifier
from typing import Dict
from models.BaseModel import BaseModel


class MLPModel(BaseModel):
    """Multi-Layer Perceptron model implementation"""

    def __init__(self):
        super().__init__("MLP")

    def define_hyperparameter_space(self) -> Dict[str, Dict]:
        return {
            "hidden_layer_sizes": {
                "type": "categorical",
                "values": [(50,), (100,), (50, 50), (100, 50)],
                "default": (100,),
            },
            "alpha": {"type": "float", "min": 0.0001, "max": 0.1, "default": 0.001},
            "learning_rate_init": {
                "type": "float",
                "min": 0.001,
                "max": 0.1,
                "default": 0.001,
            },
            "max_iter": {"type": "int", "min": 100, "max": 500, "default": 200},
        }

    def create_model(self, hyperparameters: Dict):
        self.model = MLPClassifier(
            hidden_layer_sizes=hyperparameters["hidden_layer_sizes"],
            alpha=hyperparameters["alpha"],
            learning_rate_init=hyperparameters["learning_rate_init"],
            max_iter=int(hyperparameters["max_iter"]),
            random_state=42,
        )
        return self.model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
