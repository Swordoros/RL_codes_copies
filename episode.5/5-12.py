#SARSA(lambda)算法agent

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings("ignore")

env=gym.make('Taxi-v3')
class SARSA_Agent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n=env.action_space.n
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self,state):
        if np.random.uniform() > self.epsilon:
            action=self.Q[state].argmax()
        else:
            action=np.random.randint(0, self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done,next_action):
        u=reward+self.gamma*self.Q[next_state][next_action]*(1.0-done)
        td_error=u-self.Q[state][action]
        self.Q[state][action]+=self.learning_rate*td_error


class SARSA_lambda_Agent(SARSA_Agent):
    def __init__(self, env, lambd=0.5, beta=1., gamma=0.9, learning_rate=0.1, epsilon=0.01):
        super().__init__(env, gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
        self.lambd = lambd
        self.beta = beta
        self.e = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, state, action, reward, next_state, done, next_action):
        self.e *= self.lambd * self.gamma
        self.e[state][action] += 1. + self.beta * self.e[state][action]

        u = reward + self.gamma * self.Q[next_state][next_action] * (1.0 - done)
        td_error = u - self.Q[state][action]
        self.Q+= self.learning_rate * td_error * self.e
        if done:
            self.e *= 0.0


agent = SARSA_lambda_Agent(env)

def play_SARSA(env, agent, train=False,render=False):
    episode_reward=0
    observation=env.reset()
    action=agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, done, info = env.step(action)
        episode_reward+=reward
        next_action=agent.decide(next_observation)
        if train:
            agent.learn(observation, action, reward, next_observation, done, next_action)
        if done:
            break
        observation=next_observation
        action=next_action
    return episode_reward



episodes=5000
episode_rewards=[]
for i in range(episodes):
    episode_reward=play_SARSA(env, agent, train=True)
    episode_rewards.append(episode_reward)
    if i % 100 == 0:
        print(f"Episode {i}: {episode_reward}")


plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.savefig("5-12.png")
plt.show()
plt.close()

#测试
agent.epsilon=0.0
episode_rewards=[play_SARSA(env,agent) for _ in range(100)]
print("平均回合奖励：{}/{} = {}".format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))

