#导入环境并查看观测空间和动作空间

import gym
import warnings
warnings.filterwarnings("ignore")

env=gym.make('MountainCar-v0')
print('观测空间: = {}'.format(env.observation_space))
print('动作空间: = {}'.format(env.action_space))
print('观测空间的维度: = {}'.format(env.observation_space.shape[0]))
print('动作空间的维度: = {}'.format(env.action_space.n))
print('观测范围 = {} ~ {}'.format(env.observation_space.low,env.observation_space.high))