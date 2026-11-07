import torch
import torch.nn as nn


class TorchMLP(nn.Module):
    """PyTorch reference MLP matching NumPy architecture and He initialization."""

    def __init__(self, input_size: int = 4, hidden_size: int = 64, output_size: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Match He initialization from NumPy implementation
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))

    def load_from_numpy(self, weights: dict):
        # PyTorch linear weights are transposed: (out_features, in_features)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(weights["W1"].T))
            self.fc1.bias.copy_(torch.from_numpy(weights["b1"].squeeze()))
            self.fc2.weight.copy_(torch.from_numpy(weights["W2"].T))
            self.fc2.bias.copy_(torch.from_numpy(weights["b2"].squeeze()))

    def to_numpy_weights(self) -> dict:
        return {
            "W1": self.fc1.weight.detach().cpu().numpy().T,
            "b1": self.fc1.bias.detach().cpu().numpy().reshape(1, -1),
            "W2": self.fc2.weight.detach().cpu().numpy().T,
            "b2": self.fc2.bias.detach().cpu().numpy().reshape(1, -1),
        }
