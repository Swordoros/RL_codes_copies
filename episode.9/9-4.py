#深度确定性策略梯度算法DDPG的智能体

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

#DQN深度Q网络 经验回放（6-6）
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


#OU过程
class OrnsteinUhlenbeckProcess:
    def __init__(self, size, mu=0.0, sigma=1.0, theta=0.15, dt=0.01):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.size = size

    def reset(self, x=0.0):
        self.x = x * np.ones(self.size)

    def __call__(self):
        n=np.random.normal(size=self.size)
        self.x += (self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * n)
        return self.x


#深度确定性策略梯度算法智能体
class DDPGAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs, replayer_capacity=100000, replayer_initial_transitions=10000,
                 gamma=0.99, batches=1, batch_size=64, net_learning_rate=0.005,
                 noise_salce=0.1, exploration_rate=1.0, explore=True):
        observation_dim=env.observation_space.shape[0]
        action_dim=env.action_space.shape[0]
        self.env=env
        observation_action_dim=observation_dim+action_dim
        self.action_low=env.action_space.low
        self.action_high=env.action_space.high
        self.gamma=gamma
        self.net_learning_rate=net_learning_rate
        self.explore=explore

        self.batches=batches
        self.batch_size=batch_size
        self.replayer=DQNReplayer(replayer_capacity)
        self.replayer_initial_transitions=replayer_initial_transitions

        self.noise=OrnsteinUhlenbeckProcess(size=(action_dim,), sigma=noise_salce)
        self.noise.reset()

        self.actor_evaluate_net=self.build_network(input_size=observation_dim,  **actor_kwargs)
        self.actor_target_net=self.build_network(input_size=observation_dim, **actor_kwargs)
        self.critic_evaluate_net=self.build_network(input_size=observation_action_dim, **critic_kwargs)
        self.critic_target_net=self.build_network(input_size=observation_action_dim, **critic_kwargs)

        self.update_target_net(self.actor_target_net, self.actor_evaluate_net)
        self.update_target_net(self.critic_target_net, self.critic_evaluate_net)

    #更新目标网络
    def update_target_net(self, target_net, evaluate_net, learning_rate=1.0):
        target_weights=target_net.get_weights()
        evaluate_weights=evaluate_net.get_weights()
        average_weights=[(1.0-learning_rate)*t + learning_rate*e for t, e in zip(target_weights, evaluate_weights)]
        target_net.set_weights(average_weights)

    #构建网络
    def build_network(self, input_size, hidden_sizes, output_size=1, activation=tf.nn.relu, output_activation=None, loss=keras.losses.mse, learning_rate=None):
        model=keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs={'input_shape': (input_size,)}
            model.add(keras.layers.Dense(units=hidden_size, activation=activation, kernel_initializer=GlorotUniform(seed=0), **kwargs))
        model.add(keras.layers.Dense(units=output_size, activation=output_activation, kernel_initializer=GlorotUniform(seed=0),))
        optimizer=Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    #选择动作
    def decide(self, observation):
        if self.explore and self.replayer.count < self.replayer_initial_transitions:
            return np.random.uniform(self.action_low, self.action_high)

        action=self.actor_evaluate_net.predict(observation[np.newaxis])[0]
        if self.explore:
            noise=self.noise()
            action=np.clip(action+noise, self.action_low, self.action_high)
        return action


    #训练网络
    def learn(self, observation, action, reward, next_observation, done):
        #存储经验
        self.replayer.store(observation, action, reward, next_observation, done)

        if self.replayer.count >= self.replayer_initial_transitions:
            if done:
                self.noise.reset()
            for batch in range(self.batches):
                observations, actions, rewards, next_observations, dones=self.replayer.sample(self.batch_size)

                #训练执行者网络
                observation_tensor=tf.convert_to_tensor(observations, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    actions_tensor=self.actor_evaluate_net(observation_tensor)
                    input_tensor=tf.concat([observation_tensor, actions_tensor], axis=1)
                    q_tensor=self.critic_evaluate_net(input_tensor)
                    loss_tensor= -tf.reduce_mean(q_tensor)
                grad_tensors=tape.gradient(loss_tensor, self.actor_evaluate_net.variables)
                self.actor_evaluate_net.optimizer.apply_gradients(zip(grad_tensors, self.actor_evaluate_net.variables))

                #训练评论者网络
                next_actions=self.actor_target_net.predict(next_observations)
                observation_actions=np.hstack([observations, actions])
                next_observation_actions=np.hstack([next_observations, next_actions])
                next_qs=self.critic_target_net.predict(next_observation_actions)[:, 0]
                targets=rewards + (1.0-dones) * self.gamma * next_qs
                self.critic_evaluate_net.fit(observation_actions, targets, verbose=0)

                self.update_target_net(self.actor_target_net, self.actor_evaluate_net, self.net_learning_rate)
                self.update_target_net(self.critic_target_net, self.critic_evaluate_net, self.net_learning_rate)

class TD3Agent(DDPGAgent):
    def __init__(self, env, actor_kwargs, critic_kwargs,
                 replayer_capacity=100000, replayer_initial_transitions=10000,
                 gamma=0.99, batches=1, batch_size=64,
                 net_learning_rate=0.005, noise_scale=0.1, explore=True):
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.env = env
        observation_action_dim = observation_dim + action_dim
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.gamma = gamma
        self.net_learning_rate = net_learning_rate
        self.explore = explore

        self.batches = batches
        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)
        self.replayer_initial_transitions = replayer_initial_transitions

        self.noise = OrnsteinUhlenbeckProcess(size=(action_dim,), sigma=noise_scale)
        self.noise.reset()

        self.actor_evaluate_net = self.build_network(input_size=observation_dim, **actor_kwargs)
        self.actor_target_net = self.build_network(input_size=observation_dim, **actor_kwargs)
        self.critic0_evaluate_net = self.build_network(input_size=observation_action_dim, **critic_kwargs)
        self.critic0_target_net = self.build_network(input_size=observation_action_dim, **critic_kwargs)
        self.critic1_evaluate_net = self.build_network(input_size=observation_action_dim, **critic_kwargs)
        self.critic1_target_net = self.build_network(input_size=observation_action_dim, **critic_kwargs)

        self.update_target_net(self.actor_target_net, self.actor_evaluate_net)
        self.update_target_net(self.critic0_target_net, self.critic0_evaluate_net)
        self.update_target_net(self.critic1_target_net, self.critic1_evaluate_net)

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation, done)

        if self.replayer.count >= self.replayer_initial_transitions:
            if done:
                self.noise.reset()
            for batch in range(self.batches):
                observations, actions, rewards, next_observations, dones = self.replayer.sample(self.batch_size)

                #训练执行者网络
                observation_tensor = tf.convert_to_tensor(observations, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    actions_tensor = self.actor_evaluate_net(observation_tensor)
                    input_tensor = tf.concat([observation_tensor, actions_tensor], axis=1)
                    q_tensor = self.critic0_evaluate_net(input_tensor)
                    loss_tensor = -tf.reduce_mean(q_tensor)
                grad_tensors = tape.gradient(loss_tensor, self.actor_evaluate_net.variables)
                self.actor_evaluate_net.optimizer.apply_gradients(zip(grad_tensors, self.actor_evaluate_net.variables))

                #训练评论者网络
                next_actions = self.actor_target_net.predict(next_observations)
                observation_actions = np.hstack([observations, actions])
                next_observation_actions = np.hstack([next_observations, next_actions])
                next_q0s = self.critic0_target_net.predict(next_observation_actions)[:, 0]
                next_q1s = self.critic1_target_net.predict(next_observation_actions)[:, 0]
                next_qs = np.minimum(next_q0s, next_q1s)
                targets = rewards + (1.0 - dones) * self.gamma * next_qs
                self.critic0_evaluate_net.fit(observation_actions, targets[:,np.newaxis], verbose=0)
                self.critic1_evaluate_net.fit(observation_actions, targets[:,np.newaxis], verbose=0)

                #更新目标网络
                self.update_target_net(self.actor_target_net, self.actor_evaluate_net, self.net_learning_rate)
                self.update_target_net(self.critic0_target_net, self.critic0_evaluate_net, self.net_learning_rate)
                self.update_target_net(self.critic1_target_net, self.critic1_evaluate_net, self.net_learning_rate)


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

env=gym.make('Pendulum-v1')


actor_kwargs={'hidden_sizes': [32, 64], 'learning_rate': 0.0001}
critic_kwargs={'hidden_sizes': [64, 128], 'learning_rate': 0.001}
agent=TD3Agent(env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs)

episodes=5000
episode_rewards=[]
for i in range(episodes):
    episode_reward=play_qlearning(env, agent, train=True, render=False)  #render
    episode_rewards.append(episode_reward)
    if i % 100 == 0:
        print(f"Episode {i}: {episode_reward}")

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.savefig("9-4.png")
plt.show()
plt.close()

agent.explore=False
episode_rewards=[play_qlearning(env,agent) for _ in range(100)]
print('平均回合奖励={}/{} = {}'.format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))

