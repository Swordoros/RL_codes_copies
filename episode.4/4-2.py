#从观测到状态

import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")

env=gym.make('Blackjack-v1')

def ob2state(observation):
    return (observation[0],observation[1],int(observation[2]))
