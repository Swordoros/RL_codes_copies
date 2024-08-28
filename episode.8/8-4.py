# 领近策略优化的经验回放类

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

env = gym.make('Acrobot-v1')

class PPOReplayer:
    def __init__(self):
        self.memory = pd. DataFrame()

    def store(self,df):
        self.memory = pd.concat([self.memory,df], ignore_index=True)

    def sample(self, size):
        indices = np.random.choice(self.memory.shape[0], size=size)
        return (np.stack(self.memory.loc[indices,  field]) for field in self.memory.columns)