# 重要性采样回合策略梯度算法的训练

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.initializers import GlorotUniform

warnings.filterwarnings("ignore")

env = gym.make('CartPole-v0')



class VPG_Agent:
    def __init__(self, env, policy_kwargs, baseline_kwargs=None, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.trajectory = []  # 保存轨迹
        self.policy_net = self.build_network(output_size=self.action_n, output_actication=tf.nn.softmax,
                                             loss=keras.losses.categorical_crossentropy, **policy_kwargs)
        if baseline_kwargs:
            self.baseline_net = self.build_network(output_size=1, **baseline_kwargs)

    def build_network(self, hidden_sizes, output_size, activation=tf.nn.relu, output_actication=None,
                      loss=keras.losses.mse, learning_rate=0.01):
        model = keras.Sequential()
        for hiddens in hidden_sizes:
            model.add(keras.layers.Dense(units=hiddens, activation=activation))
        model.add(keras.layers.Dense(units=output_size, activation=output_actication))
        optimizer = keras.optimizers.Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def decide(self, observation):
        probs = self.policy_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, done):
        self.trajectory.append((observation, action, reward))

        if done:
            df = pd.DataFrame(self.trajectory, columns=['observation', 'action', 'reward'])
            df['discount'] = self.gamma ** df.index.to_series()
            df['discounted_reward'] = df['reward'] * df['discount']
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()
            df['psi'] = df['discounted_return']

            x = np.stack(df['observation'])
            if hasattr(self, 'baseline_net'):
                df['baseline'] = self.baseline_net.predict(x)
                df['psi'] -= (df['baseline'] * df['discount'])
                df['return'] = df['discounted_return'] / df['discount']
                y = df['return'].values[:, np.newaxis]
                self.baseline_net.fit(x, y, verbose=0)

            y = np.eye(self.action_n)[df['action']] * df['psi'].values[:, np.newaxis]
            self.policy_net.fit(x, y, verbose=0)
            self.trajectory = []

class Off_Policy_VPG_Agent(VPG_Agent):
    def __init__(self, env, policy_kwargs, baseline_kwargs=None, gamma=0.99):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.trajectory = []  # 保存轨迹

        def dot(y_true, y_pred):
            return -tf.reduce_sum(y_true * y_pred, axis=-1)

        self.policy_net = self.build_network(output_size=self.action_n, output_actication=tf.nn.softmax,loss=dot, **policy_kwargs)
        if baseline_kwargs:
            self.baseline_net = self.build_network(output_size=1, **baseline_kwargs)

    def learn(self, observation, action, behavior, reward, done):
        self.trajectory.append((observation, action, behavior, reward))

        if done:
            df=pd.DataFrame(self.trajectory, columns=['observation', 'action', 'behavior', 'reward'])
            df['discount'] = self.gamma ** df.index.to_series()
            df['discounted_reward'] = df['reward'] * df['discount']
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()
            df['psi'] = df['discounted_return']

            x = np.stack(df['observation'])
            if hasattr(self, 'baseline_net'):
                df['baseline'] = self.baseline_net.predict(x)
                df['psi'] -= (df['baseline'] * df['discount'])
                df['return'] = df['discounted_return'] / df['discount']
                y = df['return'].values[:, np.newaxis]
                self.baseline_net.fit(x, y, verbose=0)

            y = df['behavior'].values[:, np.newaxis]
            self.policy_net.fit(x, y, verbose=0)
            self.trajectory = []



policy_kwargs={'hidden_sizes':[10,],
               'activation':tf.nn.relu,
               'learning_rate':0.01
               }
agent=Off_Policy_VPG_Agent(env, policy_kwargs=policy_kwargs)


class RandomAgent:
    def __init__(self, action_space):
        self.action_n = env.action_space.n

    def decide(self, observation):
        action = np.random.choice(self.action_n)
        behavior = 1. / self.action_n
        return action, behavior


behavior_agent = RandomAgent(env)


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
            agent.learn(observation, action, reward, done)

        observation = next_observation

        if done:
            break

    return episode_reward


episode=500
episode_rewards = []
for i in range(episode):
    observation = env.reset()
    episode_reward=0
    while True:
        action, behavior = behavior_agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.learn(observation, action, behavior, reward, done)
        if done:
            break
        observation = next_observation

    episode_reward=play_montecarlo(env, agent, render=True)
    episode_rewards.append(episode_reward)

plt.plot(episode_rewards)
plt.show()