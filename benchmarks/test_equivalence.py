import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from src.numpy_dqn.model import NumpyMLP
from src.torch_dqn.model import TorchMLP


def test_gradient_equivalence():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size = 32
    input_size = 4
    hidden_size = 64
    output_size = 2

    numpy_model = NumpyMLP(input_size, hidden_size, output_size)
    torch_model = TorchMLP(input_size, hidden_size, output_size)

    # Initialize both networks with identical parameters
    weights = numpy_model.get_weights()
    torch_model.load_from_numpy(weights)

    # Synthetic batch of states and training targets
    states = np.random.randn(batch_size, input_size).astype(np.float32)
    targets = np.random.randn(batch_size, output_size).astype(np.float32)

    # Forward pass verification
    numpy_out = numpy_model.forward(states)

    states_t = torch.from_numpy(states).float()
    states_t.requires_grad_(False)
    torch_out = torch_model(states_t)

    forward_diff = np.max(np.abs(numpy_out - torch_out.detach().numpy()))
    assert forward_diff < 1e-6, f"Forward mismatch: {forward_diff}"

    # Output error: delta = (Y - targets)
    error_out = numpy_out - targets

    # Analytical backpropagation in NumPy
    w2_grad_np = np.dot(numpy_model.A1.T, error_out) / batch_size
    b2_grad_np = np.sum(error_out, axis=0, keepdims=True) / batch_size

    hidden_error = np.dot(error_out, numpy_model.W2.T) * numpy_model.relu_derivative(numpy_model.Z1)
    w1_grad_np = np.dot(states.T, hidden_error) / batch_size
    b1_grad_np = np.sum(hidden_error, axis=0, keepdims=True) / batch_size

    # Autograd backpropagation in PyTorch
    targets_t = torch.from_numpy(targets).float()
    loss_torch = 0.5 * nn.functional.mse_loss(torch_out, targets_t, reduction="mean") * output_size
    torch_model.zero_grad()
    loss_torch.backward()

    w1_grad_pt = torch_model.fc1.weight.grad.detach().numpy().T
    b1_grad_pt = torch_model.fc1.bias.grad.detach().numpy().reshape(1, -1)
    w2_grad_pt = torch_model.fc2.weight.grad.detach().numpy().T
    b2_grad_pt = torch_model.fc2.bias.grad.detach().numpy().reshape(1, -1)

    diffs = {
        "W1 (Input -> Hidden)": np.max(np.abs(w1_grad_np - w1_grad_pt)),
        "b1 (Hidden Bias)": np.max(np.abs(b1_grad_np - b1_grad_pt)),
        "W2 (Hidden -> Output)": np.max(np.abs(w2_grad_np - w2_grad_pt)),
        "b2 (Output Bias)": np.max(np.abs(b2_grad_np - b2_grad_pt)),
    }

    print("\n" + "=" * 65)
    print("      MATHEMATICAL EQUIVALENCE TEST: NUMPY vs. PYTORCH")
    print("=" * 65)
    print(f"Forward Pass Max Difference: {forward_diff:.2e} [OK]")
    print("-" * 65)
    print(f"{'Parameter':<26} | {'Max Difference':<18} | {'Status':<10}")
    print("-" * 65)

    all_passed = True
    for name, diff in diffs.items():
        passed = diff < 1e-6
        all_passed = all_passed and passed
        status = "PASSED ✅" if passed else "FAILED ❌"
        print(f"{name:<26} | {diff:<18.2e} | {status:<10}")

    print("=" * 65)
    assert all_passed, "Gradients diverged beyond numerical tolerance threshold!"
    print("Verdict: Analytical NumPy gradients are 100% equivalent to PyTorch autograd.\n")


if __name__ == "__main__":
    test_gradient_equivalence()
