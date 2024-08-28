#随机策略一个回合21点
import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")

env=gym.make('Blackjack-v1')

observation=env.reset()
print('观测 = {}'.format(observation))

while True:
    print('玩家 = {}  庄家 = {}'.format(env.player, env.dealer))
    action=np.random.choice(env.action_space.n)
    print('动作 = {}'.format(action))
    observation,reward,done,info=env.step(action)
    print(env.step(action))
    print('观测 = {}, 奖励 = {}, 是否结束 = {}'.format(observation, reward, done))

    if done:
        print('游戏结束')
        break

