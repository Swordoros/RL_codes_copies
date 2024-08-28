#雅达利游戏bot NatureDQN算法

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")


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

env=gym.make('Breakout-v0')

class AtariAgent:
    def __init__(self, env, input_shape, learning_rate=0.00025,load_path=None, gamma=0.99, replay_memory_size=1000000,
                 batch_size=32, replay_start_size=0, epsilon=1.0, epsilon_decrease_rate=9e-7, min_epsilon=0.1,
                 random_inital_steps=0, clip_reward=True, rescale_state=True, update_freq=1, target_network_update_freq=1):

        self.action_n=env.action_space.n
        self.gamma=gamma

        #经验回放参数
        self.replay_memory_size=replay_memory_size
        self.replay_start_size=replay_start_size
        self.batch_size=batch_size
        self.replayer=DQNReplayer(self.replay_memory_size)

        #

