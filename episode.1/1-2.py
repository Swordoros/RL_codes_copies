#根据指定确定性策略决定动作的智能体

import gym
import warnings
warnings.filterwarnings("ignore")

env=gym.make('MountainCar-v0')
class BespokeAgent:
    def __init__(self, env):
        pass

    def decide(self, obervation): #决策
        position, velocity = obervation
        lb=min(-0.09 * (position + 0.25)**2 + 0.03, 0.3 * (position + 0.9)**4 - 0.008)
        ub=-0.07 * (position + 0.38)**2 + 0.06
        if lb<velocity<ub:
            action=2
        else:
            action=0
        return action

    def learn(self, *args): #学习
        pass

agent = BespokeAgent(env)