#导入小车上山环境

import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")

env=gym.make('MountainCar-v0')
env=env.unwrapped

print('观测空间： {}'.format(env.observation_space))
print('动作空间： {}'.format(env.action_space))
print('位置范围： {}'.format((env.min_position, env.max_position)))
print('速度范围： {}'.format((-env.max_speed, env.max_speed)))
print('目标位置： {}'.format(env.goal_position))
