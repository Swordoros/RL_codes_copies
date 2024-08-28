# OU过程(Ornstein-Uhlenbeck Process)的实现

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


class OrnsteinUhlenbeckProcess:
    def __init__(self, size, mu=0.0, sigma=1.0, theta=0.15, dt=0.01):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.size = size

    def reset(self, x=0.0):
        self.x = x * np.ones(self.size)

    def __call__(self):
        n=np.random.normal(size=self.size)
        self.x += (self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * n)
        return self.x
