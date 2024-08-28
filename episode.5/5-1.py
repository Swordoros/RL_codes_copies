#初始化gym并且运行一步taxi-v3

import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")

env=gym.make('Taxi-v3')
state=env.reset()
taxirow, taxicol, passloc, destidx = env.unwrapped.decode(state)
print(taxirow, taxicol, passloc, destidx)
print('出租车位置 = {}'.format((taxirow,taxicol)))
print('乘客位置 = {}'.format(env.unwrapped.locs[passloc]))
print('目的地索引 = {}'.format(env.unwrapped.locs[destidx]))
env.render()
env.step(0)

import time
time.sleep(3)
env.close()
