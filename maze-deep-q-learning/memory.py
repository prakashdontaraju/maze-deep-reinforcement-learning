import numpy as np
from queue import deque

class Memory():
    """Sets up the memory element"""

    def __init__(self, max_size):
        """Initializes the memory element"""
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        """Adds player experience to the memory element"""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Collects player experience from the memory element"""

        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)

        return [self.buffer[i] for i in index]