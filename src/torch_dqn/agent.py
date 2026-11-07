import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.common.memory import ReplayMemory
from src.torch_dqn.model import TorchMLP


class TorchDQNAgent:
    """PyTorch reference DQN agent matching NumPy agent interface."""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        output_size: int = 2,
        memory_capacity: int = 10000,
        epsilon_start: float = 1.0,
        device: str = "cpu",
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = torch.device(device)

        self.model = TorchMLP(input_size, hidden_size, output_size).to(self.device)
        self.target_model = TorchMLP(input_size, hidden_size, output_size).to(self.device)
        self.memory = ReplayMemory(capacity=memory_capacity)
        self.epsilon = epsilon_start

        # Use standard SGD to remain directly comparable to NumPy
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.00025)
        self.loss_fn = nn.MSELoss()
        self.update_target_network()

    def update_target_network(self):
        # Synchronize target network weights
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.output_size))
        with torch.no_grad():
            state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
            q_vals = self.model(state_tensor)
            return int(torch.argmax(q_vals).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self, batch_size: int = 64, gamma: float = 0.99, lr: float = 0.00025) -> float:
        if len(self.memory) < batch_size:
            return 0.0

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.vstack(states), dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.vstack(next_states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)

        # Gather Q-values for executed actions
        current_q = self.model(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q = self.target_model(next_states_t).max(1)[0]
            targets = rewards_t + (~dones_t) * gamma * next_q

        loss = self.loss_fn(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradient values to match [-1.0, 1.0] in NumPy
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
        self.optimizer.step()

        return float(loss.item())

    def save(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))
        self.update_target_network()
