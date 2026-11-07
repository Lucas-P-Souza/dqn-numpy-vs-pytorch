import numpy as np

from src.common.memory import ReplayMemory
from src.numpy_dqn.model import NumpyMLP


class NumpyDQNAgent:
    """Deep Q-Network agent with analytical backpropagation in NumPy."""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        output_size: int = 2,
        memory_capacity: int = 10000,
        epsilon_start: float = 1.0,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = NumpyMLP(input_size, hidden_size, output_size)
        self.target_model = NumpyMLP(input_size, hidden_size, output_size)
        self.memory = ReplayMemory(capacity=memory_capacity)
        self.epsilon = epsilon_start
        self.update_target_network()

    def update_target_network(self):
        # Sync target network parameters with policy network
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.output_size))
        q_vals = self.model.forward(state.reshape(1, -1))
        return int(np.argmax(q_vals))

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self, batch_size: int = 64, gamma: float = 0.99, lr: float = 0.00025) -> float:
        if len(self.memory) < batch_size:
            return 0.0

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.vstack(states)
        next_states = np.vstack(next_states)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)

        current_q = self.model.forward(states)
        next_q = self.target_model.forward(next_states)
        max_next_q = np.max(next_q, axis=1)

        # Standard Bellman target update: r + gamma * max Q(s', a') * (1 - done)
        targets = current_q.copy()
        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += gamma * max_next_q[i]
            targets[i, actions[i]] = target

        td_error = current_q - targets
        loss = float(np.mean(td_error**2))

        # Output layer gradients
        w2_grad = np.dot(self.model.A1.T, td_error) / batch_size
        b2_grad = np.sum(td_error, axis=0, keepdims=True) / batch_size

        # Hidden layer gradients propagated through ReLU derivative
        hidden_error = np.dot(td_error, self.model.W2.T) * self.model.relu_derivative(self.model.Z1)
        w1_grad = np.dot(states.T, hidden_error) / batch_size
        b1_grad = np.sum(hidden_error, axis=0, keepdims=True) / batch_size

        # Gradient clipping prevents exploding updates
        for grad in (w1_grad, b1_grad, w2_grad, b2_grad):
            np.clip(grad, -1.0, 1.0, out=grad)

        # Parameter update via SGD
        self.model.W1 -= lr * w1_grad
        self.model.b1 -= lr * b1_grad
        self.model.W2 -= lr * w2_grad
        self.model.b2 -= lr * b2_grad

        return loss

    def save(self, filepath: str):
        np.savez(filepath, **self.model.get_weights())

    def load(self, filepath: str):
        data = np.load(filepath)
        weights = {k: data[k] for k in data.files}
        self.model.set_weights(weights)
        self.update_target_network()
