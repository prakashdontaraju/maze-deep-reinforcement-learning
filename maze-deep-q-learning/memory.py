import numpy as np
from queue import deque

class Memory():
    """sets up the memory element"""

    def __init__(self, max_size):
        """initializes the memory element"""
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        """adds player experience to the memory element"""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """collects player experience from the memory element"""

        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)

        return [self.buffer[i] for i in index]
