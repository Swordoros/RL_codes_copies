#随机测率agent

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.initializers import GlorotUniform

warnings.filterwarnings("ignore")

env = gym.make('CartPole-v0')


class RandomAgent:
    def __init__(self, action_space):
        self.action_n = action_space

    def decide(self, observation):
        action=np.random.choice(self.action_n)
        behavior=1. /self.action_n
        return action, behavior

behavior_agent=RandomAgent(env)
