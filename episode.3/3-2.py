#求随即策略的期望奖励

import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")

env=gym.make('FrozenLake-v1')
env=env.unwrapped

def play_policy(env, policy,render=False):
    total_reward=0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action=np.random.choice(env.action_space.n, p=policy[observation])
        observation, reward, done, Placeholder, info= env.step(action)
        total_reward+=reward

        if done:
            break

    return total_reward

random_policy=np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
print(random_policy)
episode_rewards=[play_policy(env, random_policy) for _ in range(1000)]
print('随机策略 平均奖励 = {}'.format(np.mean(episode_rewards)))
