import numpy as np


class NumpyMLP:
    """Two-layer MLP Q-network implemented from scratch in NumPy."""

    def __init__(self, input_size: int = 4, hidden_size: int = 64, output_size: int = 2):
        # He initialization preserves activation variance across ReLU layers
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # Cached intermediate activations for backpropagation
        self.Z1 = None
        self.A1 = None
        self.Z2 = None

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        # Step derivative of ReLU activation
        return (x > 0.0).astype(float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.Z1 = np.dot(x, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2

    def get_weights(self) -> dict:
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def set_weights(self, weights: dict):
        self.W1 = weights["W1"].copy()
        self.b1 = weights["b1"].copy()
        self.W2 = weights["W2"].copy()
        self.b2 = weights["b2"].copy()
