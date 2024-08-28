# 领近策略优化算法的agent

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


class PPOReplayer:
    def __init__(self):
        self.memory = pd.DataFrame()

    def store(self, df):
        self.memory = pd.concat([self.memory, df], ignore_index=True)

    def sample(self, size):
        indices = np.random.choice(self.memory.shape[0], size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)

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

class PPOAgent(QActorCriticAgent):
    def __init__(self,env, actor_kwargs, critic_kwargs, clip_ratio=0.1, gamma=0.99, lambd=0.99, min_trajectory_length=1000, batches=1,batch_size=64):
        self.action_n=env.action_space.n
        self.gamma=gamma
        self.lambd=lambd
        self.min_trajectory_length=min_trajectory_length
        self.batches=batches
        self.batch_size=batch_size
        self.trajectory=[]
        self.replayer=PPOReplayer()

        def ppo_loss(y_true, y_pred):
            p=y_pred
            p_old=y_true[:, :self.action_n]
            advantage= y_true[:, self.action_n:]
            surrogate_advantage = (p/p_old)*advantage
            clip_times_advantage = clip_ratio * advantage
            max_surrogate_advantage = advantage + tf.where(advantage > 0. , clip_times_advantage, -clip_times_advantage)
            clipped_surrogate_advantage = tf.minimum(surrogate_advantage, max_surrogate_advantage)
            return -tf.reduce_mean(clipped_surrogate_advantage, axis=-1)

        self.actor_net=self.build_network(output_size=self.action_n, output_activation=tf.nn.softmax, loss=ppo_loss,**actor_kwargs)
        self.critic_net=self.build_network(output_size=1, **critic_kwargs)

    def learn(self, observation, action, reward, done):
        self.trajectory.append((observation, action, reward))

        if done:
            df=pd.DataFrame(self.trajectory, columns=['observation', 'action', 'reward'])
            observations = np.stack(df['observation'])
            df['v']=self.critic_net.predict(observations)
            psi=self.actor_net.predict(observations)
            df['pi']=[a.flatten() for a in np.split(psi, psi.shape[0])]
            df['next_v']=df['v'].shift(-1).fillna(0.)
            df['u']=df['reward']+self.gamma*df['next_v']
            df['delta']=df['u']-df['v']
            df['return']=df['reward']
            df['advantage']=df['delta']
            for i in df.index[-2::1]:
                df.loc[i,'return'] +=self.gamma*df.loc[i+1,'return']
                df.loc[i,'advantage'] += self.gamma*self.lambd*df.loc[i+1,'advantage']
            fields=['observation','action', 'pi', 'advantage', 'return']
            self.replayer.store(df[fields])
            self.trajectory=[]
            if len(self.replayer.memory)>self.min_trajectory_length:
                for batch in range(self.batches):
                    observations, actions, pis, advantages, returns=self.replayer.sample(self.batch_size)
                    ext_advantages=np.zeros_like(pis)
                    ext_advantages[range(self.batch_size), actions]=advantages
                    actor_targets=np.hatack([pis, ext_advantages])
                    self.actor_net.fit(observations, actor_targets, verbose=0)
                    self.critic_net.fit(observations, returns, verbose=0)

                self.replayer=PPOReplayer()


actor_kwargs={'hidden_sizes' : [100,],
              'learning_rate' : 0.001}
critic_kwargs={'hidden_sizes' : [100,],
               'learning_rate' : 0.001}
agent=PPOAgent(env, actor_kwargs, critic_kwargs)

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


episodes=500
episode_rewards=[]
for episode in range(episodes):
    episode_reward=play_montecarlo(env, agent , render=True, train=True)
    episode_rewards.append(episode_reward)
plt.plot(episode_rewards)
plt.show()