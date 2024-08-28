#CliffWalking运行一个回合

import gym
import warnings
warnings.filterwarnings("ignore")

env=gym.make('CliffWalking-v0')

import numpy as np
def play_once(env, policy):
    total_reward = 0
    state = env.reset()
    while True:
        loc=np.unravel_index(state,env.shape)
        print('状态={}，位置={}'.format(state,loc),end=' ')
        action=np.random.choice(env.nA, p=policy[state])
        state, reward, done, _ = env.step(action)
        print('动作={}，奖励={}'.format(action,reward))
        total_reward += reward
        if done:
            break
    return total_reward