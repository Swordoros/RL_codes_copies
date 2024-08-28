#导入CliffWalking环境

import gym
import warnings
warnings.filterwarnings("ignore")

env=gym.make('CliffWalking-v0')
print('观测空间: = {}'.format(env.observation_space))
print('动作空间: = {}'.format(env.action_space))
print('状态数量: = {}'.format(env.nS))
print('动作数量: = {}'.format(env.nA))
print('地图大小：= {}'.format(env.shape))