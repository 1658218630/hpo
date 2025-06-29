from sklearn.neural_network import MLPClassifier
from typing import Dict
from models.BaseModel import BaseModel


class EnhancedMLPModel(BaseModel):
    """
    Enhanced MLP with wide hyperparameter ranges for clear performance differences.
    Minimal implementation - only what's needed for the tournament.
    """

    def __init__(self):
        super().__init__("EnhancedMLP")

    def define_hyperparameter_space(self) -> Dict[str, Dict]:
        return {
            # Architecture - huge impact on performance and speed
            "hidden_layer_sizes": {
                "type": "categorical",
                "values": [
                    (32,),
                    (64,),
                    (100,),
                    (128,),  # Small networks
                    (50, 50),
                    (100, 50),
                    # (128, 64),  # Medium networks
                    # (200,),
                    # (256,),
                    # (200, 100),  # Large networks
                    # (256, 128),
                    # (300, 150),
                    # (256, 128, 64),  # Very large networks
                ],
                "default": (100,),
            },
            # Learning rate - dramatic impact on convergence
            "learning_rate_init": {
                "type": "float",
                "min": 0.0001,  # Very slow
                "max": 0.5,  # Very fast (may diverge)
                "default": 0.001,
            },
            # Regularization - affects overfitting
            "alpha": {
                "type": "float",
                "min": 0.00001,  # Almost no regularization
                "max": 1.0,  # Heavy regularization
                "default": 0.0001,
            },
            # Training time control
            "max_iter": {
                "type": "int",
                "min": 50,  # Quick but may not converge
                "max": 200,  # Long training
                "default": 200,
            },
            # Solver choice affects convergence
            "solver": {
                "type": "categorical",
                "values": ["adam", "sgd", "lbfgs"],
                "default": "adam",
            },
        }

    def create_model(self, hyperparameters: Dict):
        use_early_stopping = getattr(self, "_use_early_stopping", False)

        mlp_params = {
            "hidden_layer_sizes": hyperparameters["hidden_layer_sizes"],
            "alpha": hyperparameters["alpha"],
            "learning_rate_init": hyperparameters["learning_rate_init"],
            "max_iter": int(hyperparameters["max_iter"]),
            "solver": hyperparameters["solver"],
            "random_state": 42,
        }

        # Only add early stopping if not in CV mode
        if use_early_stopping:
            mlp_params.update(
                {
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                    "n_iter_no_change": 10,
                }
            )

        self.model = MLPClassifier(**mlp_params)
        return self.model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
