import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict
from models.BaseModel import BaseModel
from sklearn.preprocessing import LabelEncoder


class PyTorchMLP(nn.Module):
    """PyTorch MLP for GPU acceleration"""

    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.1):
        super(PyTorchMLP, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GPUMLPModel(BaseModel):
    """
    GPU-accelerated MLP using PyTorch
    Significant speedup for large datasets (>10k samples)
    """

    def __init__(self):
        super().__init__("GPU_MLP")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = LabelEncoder()  # ← WICHTIG: Immer initialisieren
        print(f"GPU MLP using device: {self.device}")

        if not torch.cuda.is_available():
            print("⚠️ CUDA not available, falling back to CPU")

    def define_hyperparameter_space(self) -> Dict[str, Dict]:
        return {
            "hidden_layer_sizes": {
                "type": "categorical",
                "values": [
                    (64,),
                    (128,),
                    (256,),  # Single layer
                    (128, 64),
                    (256, 128),  # Two layers
                    (256, 128, 64),  # Three layers
                ],
                "default": (128,),
            },
            "learning_rate": {
                "type": "float",
                "min": 0.0001,
                "max": 0.01,
                "default": 0.001,
            },
            "batch_size": {
                "type": "int",
                "min": 32,
                "max": 512,
                "default": 128,
            },
            "dropout": {
                "type": "float",
                "min": 0.0,
                "max": 0.5,
                "default": 0.1,
            },
            "max_epochs": {
                "type": "int",
                "min": 10,
                "max": 100,
                "default": 50,
            },
        }

    def create_model(self, hyperparameters: Dict):
        self.hyperparameters = hyperparameters
        # label_encoder already initialized in __init__
        wrapper = SklearnGPUMLPWrapper(self, hyperparameters)
        self.model = wrapper
        # Return sklearn-compatible wrapper
        return self.model

    def fit(self, X_train, y_train):
        # Store data for wrapper compatibility
        self._X_train = X_train
        self._y_train = y_train

        # Prepare data
        X_tensor = torch.FloatTensor(X_train).to(self.device)

        # Encode labels - label_encoder is now always available
        y_encoded = self.label_encoder.fit_transform(y_train)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)

        # Model setup
        input_size = X_train.shape[1]
        output_size = len(np.unique(y_encoded))
        hidden_sizes = self.hyperparameters["hidden_layer_sizes"]

        self.pytorch_model = PyTorchMLP(  # ← Renamed to avoid conflict
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            dropout=self.hyperparameters["dropout"],
        ).to(self.device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.pytorch_model.parameters(),  # ← Use pytorch_model
            lr=self.hyperparameters["learning_rate"],
            weight_decay=1e-5,  # L2 regularization
        )

        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, batch_size=int(self.hyperparameters["batch_size"]), shuffle=True
        )

        # Training loop with early stopping
        self.pytorch_model.train()  # ← Use pytorch_model
        best_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(int(self.hyperparameters["max_epochs"])):
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.pytorch_model(batch_X)  # ← Use pytorch_model
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)

            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Mark as fitted
        self.is_fitted = True

    def predict(self, X_test):
        if not hasattr(self, "is_fitted") or not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.pytorch_model.eval()  # ← Use pytorch_model
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            outputs = self.pytorch_model(X_tensor)  # ← Use pytorch_model
            _, predicted = torch.max(outputs.data, 1)
            predicted_cpu = predicted.cpu().numpy()

        # Decode labels back to original
        return self.label_encoder.inverse_transform(predicted_cpu)


class SklearnGPUMLPWrapper:
    """
    Sklearn-compatible wrapper for PyTorch GPU MLP
    This makes it work with cross_val_score
    """

    def __init__(self, gpu_mlp_model=None, hyperparameters=None):
        # Support both: direct construction and sklearn clone
        if gpu_mlp_model is None:
            # Sklearn clone path - create new model
            self.gpu_mlp_model = GPUMLPModel()
            self.hyperparameters = hyperparameters or {}
        else:
            # Direct construction path
            self.gpu_mlp_model = gpu_mlp_model
            self.hyperparameters = hyperparameters or {}

        self.is_fitted = False

    def fit(self, X, y):
        """Sklearn-compatible fit method"""
        # Ensure hyperparameters are set in the model
        self.gpu_mlp_model.hyperparameters = self.hyperparameters
        self.gpu_mlp_model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Sklearn-compatible predict method"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.gpu_mlp_model.predict(X)

    def score(self, X, y):
        """Sklearn-compatible score method for cross_val_score"""
        from sklearn.metrics import accuracy_score

        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def get_params(self, deep=True):
        """Sklearn-compatible get_params for cross-validation"""
        return self.hyperparameters.copy()

    def set_params(self, **params):
        """Sklearn-compatible set_params"""
        # Update hyperparameters in both wrapper and model
        self.hyperparameters.update(params)
        if hasattr(self.gpu_mlp_model, "hyperparameters"):
            self.gpu_mlp_model.hyperparameters.update(params)
        return self

    def __sklearn_clone__(self):
        """Custom sklearn clone method"""
        # Create new instance with same parameters
        return SklearnGPUMLPWrapper(hyperparameters=self.hyperparameters.copy())
