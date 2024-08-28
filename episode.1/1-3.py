#智能体和环境交互一个回合的代码

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

def play_montecarlo(env, agent, render=False, train=False):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()

        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward

        if train:
            agent.learn(observation, action, reward, next_observation, done)

        observation = next_observation

        if done:
            break

    return episode_reward

#env.seed(0)
episode_reward = play_montecarlo(env, agent, render=True)
print("Episode reward: {}".format(episode_reward))
env.close()
