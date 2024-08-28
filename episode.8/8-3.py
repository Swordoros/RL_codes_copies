# 带资格迹的A/C算法

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

env = gym.make('Acrobot-v1')

class QActorCriticAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n=env.action_space.n
        self.gamma=gamma
        self.discount=1.
        self.actor_net=self.build_network(output_size=self.action_n, output_activation=tf.nn.softmax,loss=keras.losses.categorical_crossentropy,**actor_kwargs)
        self.critic_net=self.build_network(output_size=self.action_n, **critic_kwargs)

    def build_network(self, hidden_sizes, output_size,input_size=None, activation=tf.nn.relu, output_activation=None, loss=keras.losses.mse, learning_rate=0.01):
        model=keras.Sequential()
        for idx, hidden_size in enumerate(hidden_sizes):
            kwargs={}
            if idx==0 and input_size is not None:
                kwargs['input_shape']=(input_size,)
            model.add(keras.layers.Dense(units=hidden_size, activation=activation, kernel_initializer=GlorotUniform(seed=0), **kwargs))
        model.add(keras.layers.Dense(units=output_size, activation=output_activation, kernel_initializer=GlorotUniform(seed=0)))
        optimizer=Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def decide(self,observation):
        probs = self.actor_net.predict(observation[np.newaxis])[0]
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, next_observation, done, next_action=None):
        x = observation[np.newaxis]
        u = self.critic_net.predict(x)
        q = u[0, action]
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor=self.actor_net(x_tensor)[0,action]
            logpi_tensor=tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.0))
            loss_tensor= -self.discount * q * logpi_tensor
        gard_tensors=tape.gradient(loss_tensor, self.actor_net.variables)
        self.actor_net.optimizer.apply_gradients(zip(gard_tensors, self.actor_net.variables))

        #训练critic网络
        u[0 ,action]=reward
        if not done:
            q=self.critic_net.predict(next_observation[np.newaxis])[0, 0]  #action? next_action?
            u[0 ,action]  += self.gamma * q
        self.critic_net.fit(x, u, verbose=0)

        if done:
            self.discount = 1.
        else:
            self.discount *= self.gamma


class ElibilityTraceActorCriticAgent(QActorCriticAgent):
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99, actor_lambd=0.9, critic_lambd=0.9):
        observation_dim=env.observation_space.shape[0]
        self.action_n=env.action_space.n
        self.gamma=gamma
        self.discount=1.
        self.actor_lambd=actor_lambd
        self.critic_lambd=critic_lambd

        self.actor_net=self.build_network(input_size=observation_dim, output_size=self.action_n, output_activation=tf.nn.softmax, **actor_kwargs)
        self.critic_net=self.build_network(input_size=observation_dim, output_size=self.action_n, **critic_kwargs)
        self.actor_traces=[np.zeros_like(w) for w in self.actor_net.get_weights ]
        self.critic_traces=[np.zeros_like(w) for w in self.critic_net.get_weights ]

    def learn(self, observation, action, reward, next_observation, done):
        q=self.critic_net.predict(observation[np.newaxis])[0,0]
        u=reward+(1.0-done)*self.gamma*self.critic_net.predict(next_observation[np.newaxis])[0,0]
        td_error=u-q

        x_tensor=tf.convert_to_tensor(observation[np.newaxis], dtype=tf.float32)
        with tf.GradientTape() as tape:
            pi_tensor=self.actor_net(x_tensor)
            logpi_tensor=tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.0))
            logpi_pick_tensor=logpi_tensor[0,action]
        gard_tensors=tape.gradient(logpi_pick_tensor, self.actor_net.variables)
        self.actor_traces=[self.gamma * self.actor_lambd * trace + self.discount * gard.numpy() for trace, gard in zip(self.actor_traces, gard_tensors)]
        actor_grads=[tf.convert_to_tensor(trace, dtype=tf.float32) for trace in self.actor_traces]
        actor_grads_and_vars=tuple(zip(actor_grads, self.actor_net.variables))
        self.actor_net.optimizer.apply_gradients(actor_grads_and_vars)

        with tf.GradientTape() as tape:
            v_tensor=self.critic_net(x_tensor)
        grad_tensor=tape.gradient(v_tensor, self.critic_net.variables)
        self.critic_traces=[self.gamma * self.critic_lambd * trace + self.discount * grad.numpy() for trace, grad in zip(self.critic_traces, grad_tensor)]
        critic_grads=[tf.convert_to_tensor(-td_error*trace , dtype=tf.float32) for trace in self.critic_traces]
        critic_grads_and_vars=tuple(zip(critic_grads, self.critic_net.variables))
        self.critic_net.optimizer.apply_gradients(critic_grads_and_vars)

        if done:
            self.actor_traces=[np.zeros_like(w) for w in self.actor_net.get_weights ]
            self.critic_traces=[np.zeros_like(w) for w in self.critic_net.get_weights ]
            self.discount = 1.
        else:
            self.discount *= self.gamma




actor_kwargs={'hidden_sizes' : [100,],
              'learning_rate' : 0.001}
critic_kwargs={'hidden_sizes' : [100,],
               'learning_rate' : 0.001}

agent=ElibilityTraceActorCriticAgent(env, actor_kwargs, critic_kwargs)


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
    episode_reward=play_qlearning(env, agent, train=True,render=True)
    episode_rewards.append(episode_reward)
    if i % 100 == 0:
        print(f"Episode {i}: {episode_reward}")

'''
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.savefig("8-3.png")
plt.show()
plt.close()
'''

#测试
'''
agent.epsilon=0.0
episode_rewards=[play_qlearning(env,agent) for _ in range(100)]
print("平均回合奖励：{}/{} = {}".format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))
'''
