#对小车总是施加向右的力（action=2），限制步数为1000，并绘制位置和速度的变化曲线

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

env=gym.make('MountainCar-v0')
env=env.unwrapped

positions, velcoties = [], []
observation = env.reset()
#env.render()

for i in range(1000):
    positions.append(observation[0])
    velcoties.append(observation[1])
    next_observation, reward, done, info, _ = env.step(action=2)
    if done:
        break
    observation = next_observation

if next_observation[0] >= 0.5:
    print("Success!")

else:
    print("Failure!")

fig, ax = plt.subplots()
ax.plot(positions, label='位置')
ax.plot(velcoties, label='速度')
ax.legend()
fig.savefig('6-2.png')
fig.show()

