#Q学习agent

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings("ignore")

env=gym.make('Taxi-v3')

class QLearningAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.1):
        self.gamma = gamma
        self.learner_rate = learning_rate
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
        u=reward + self.gamma * np.max(self.Q[next_state])*(1-done)
        td_error = u - self.Q[state][action]
        self.Q[state][action] += self.learner_rate * td_error

agent = QLearningAgent(env)

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
plt.savefig("5-10.png")
plt.show()
plt.close()

#测试
agent.epsilon=0.0
episode_rewards=[play_qlearning(env,agent) for _ in range(100)]
print("平均回合奖励：{}/{} = {}".format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))

pd.DataFrame(agent.Q)
policy=np.eye(agent.action_n)[agent.Q.argmax(axis=-1)]
pd.DataFrame(policy)
