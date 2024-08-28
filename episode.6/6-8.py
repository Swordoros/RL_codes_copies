#双重DQN学习agent

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.initializers import GlorotUniform

warnings.filterwarnings("ignore")

env = gym.make('MountainCar-v0')

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity), columns=['observation', 'action','reward', 'next_observation', 'done'])
        self.i=0
        self.count=0
        self.capacity=capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i+1) % self.capacity
        self.count = min(self.count+1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.count, batch_size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)



class DQNAgent:
    def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.001, replayer_capacity=10000, batch_size=64):
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)
        observation_dim = env.observation_space.shape[0]

        #评估网络
        self.evaluate_net=self.build_network(input_size=observation_dim, output_size=self.action_n, **net_kwargs)
        #目标网络
        self.target_net=self.build_network(input_size=observation_dim, output_size=self.action_n, **net_kwargs)
        self.target_net.set_weights(self.evaluate_net.get_weights())

    def build_network(self, input_size, hidden_sizes, output_size, activation=tf.nn.relu, output_activation=None, learning_rate=0.01):
        model = keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size,)) if not layer else {}
            model.add(keras.layers.Dense(units=hidden_size, activation=activation, kernel_initializer=GlorotUniform(seed=0), **kwargs))
        model.add(keras.layers.Dense(units=output_size, activation=output_activation, kernel_initializer=GlorotUniform(seed=0)))
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def learn(self, observation, action, reward, next_observation, done):
        #存储经验
        self.replayer.store(observation, action, reward, next_observation, done)
        #经验回放
        observations, actions, rewards, next_observations, dones = self.replayer.sample(self.batch_size)

        next_qs = self.target_net.predict(next_observations)
        next_max_qs = np.max(next_qs, axis=-1)
        us = rewards + (1-dones) * self.gamma * next_max_qs
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done:
            self.target_net.set_weights(self.evaluate_net.get_weights())

    def decide(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_n)
        else:
            qs = self.evaluate_net.predict(observation[np.newaxis])
            return np.argmax(qs)

class DoubleDQNAgent(DQNAgent):
    def learn(self, observation, action, reward, next_observation, done):
        #存储经验
        self.replayer.store(observation, action, reward, next_observation, done)
        #经验回放
        observations, actions, rewards, next_observations, dones = self.replayer.sample(self.batch_size)

        next_eval_qs=self.evaluate_net.predict(next_observations)
        next_actions=np.argmax(next_eval_qs, axis=-1)
        next_qs=self.target_net.predict(next_observations)
        next_max_qs=next_qs[np.arange(next_qs.shape[0]), next_actions]
        us=rewards+(1-dones)*self.gamma*next_max_qs
        targets=self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions]=us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done:
            self.target_net.set_weights(self.evaluate_net.get_weights())

net_kwargs = {'hidden_sizes':[64,],
              'learning_rate':0.01}
agent = DoubleDQNAgent(env, net_kwargs=net_kwargs)


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
    episode_reward=play_qlearning(env, agent, train=True, render=True)
    episode_rewards.append(episode_reward)
    print(f"Episode {i}: {episode_reward}")


plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.savefig("6-8.png")
plt.show()
plt.close()

#测试
agent.epsilon=0.0
episode_rewards=[play_qlearning(env,agent, render=True) for _ in range(10)]
print("平均回合奖励：{}/{} = {}".format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))
