#DQN经验回放的实现

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

env = gym.make('MountainCar-v0')

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity), columns=['observation', 'action','reward', 'next_observation', 'done'])
        self.i=0
        self.count=0
        self.capacity=capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i+1) % self.capacity
        self.count = min(self.count+1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.count, batch_size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)