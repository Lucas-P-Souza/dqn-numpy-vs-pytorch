import random
from collections import deque


class ReplayMemory:
    """Fixed-capacity buffer to decorrelate consecutive transitions."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        # Uniform sampling breaks temporal correlation
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
