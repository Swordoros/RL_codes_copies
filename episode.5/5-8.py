#训练期望SARSA算法

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings("ignore")

env=gym.make('Taxi-v3')

class Expected_SARSA_Agent:
    def __init__(self, env, gamma=0.9, learner_rate=0.1, epsilon=0.01):
        self.gamma = gamma
        self.learner_rate = learner_rate
        self.epsilon = epsilon
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.action_n=env.action_space.n

    def decide(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.action_n)
        else:
            action = np.argmax(self.Q[state])
        return action

    def learn(self, state, action, reward, next_state, done):
        v = (self.Q[next_state].sum() * self.epsilon + self.Q[next_state].max() * (1 - self.epsilon))
        u=reward+self.gamma*v*(1.0-done)
        td_error=u-self.Q[state][action]
        self.Q[state][action]+=self.learner_rate*td_error

agent=Expected_SARSA_Agent(env)

def play_qlearning(env, agent, train=False, render=False):
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
        if done:
            break
        observation = next_observation
    return episode_reward

episodes=5000
episode_rewards=[]
for i in range(episodes):
    episode_reward=play_qlearning(env, agent, train=True)
    episode_rewards.append(episode_reward)
    if i % 100 == 0:
        print(f"Episode {i}: {episode_reward}")

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.savefig("5-8.png")
plt.show()
plt.close()