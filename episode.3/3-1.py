#用策略执行一个回合

import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")

env=gym.make('FrozenLake-v1')
env=env.unwrapped

print(env.observation_space)
print(env.action_space)

print(env.unwrapped.P[13][2])

def play_policy(env, policy,render=False):
    total_reward=0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action=np.random.choice(env.action_space.n, p=policy[observation])
        observation, reward, done, info = env.step(action)
        total_reward+=reward
        if done:
            break

    return total_reward
