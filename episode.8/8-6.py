# 柔性A-C算法智能体

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



class SACAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs, replayer_capacity=10000, gamma=0.99, alpha=0.99, batches=1, batch_size=64, net_learning_rate=0.995):
        observation_dim=env.observation_space.shape[0]
        self.action_n=env.action_space.n
        self.gamma=gamma
        self.alpha=alpha
        self.batches=batches
        self.batch_size=batch_size
        self.net_learning_rate=net_learning_rate
        self.replayer=DQNReplayer(replayer_capacity)

        def sac_loss(y_true, y_pred):
            qs= alpha * tf.math.xlogy(y_true, y_pred) -y_pred*y_true
            return tf.reduce_mean(qs, axis=-1)

        self.actor_net=self.build_network(input_size=observation_dim, output_size=self.action_n, output_activation=tf.nn.softmax, loss=sac_loss, **actor_kwargs)
        self.q0_net=self.build_network(input_size=observation_dim, output_size=self.action_n, **critic_kwargs)
        self.q1_net=self.build_network(input_size=observation_dim, output_size=self.action_n, **critic_kwargs)
        self.v_evaluate_net=self.build_network(input_size=observation_dim, output_size=1, **critic_kwargs)
        self.v_target_net=self.build_network(input_size=observation_dim, output_size=1, **critic_kwargs)



    def build_network(self, hidden_sizes, input_size, output_size, activation=tf.nn.relu, output_activation=None, loss=keras.losses.mse, learning_rate=0.01):
        model=keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs={'input_shape': (input_size,)} if layer==0 else {}
            model.add(keras.layers.Dense(units=hidden_size, activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size, activation=output_activation))
        optimizer=Adam(learning_rate)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def update_target_net(self, target_net, evaluate_net, learning_rate=1.0):
        target_weights=target_net.get_weights()
        evaluate_weights=evaluate_net.get_weights()
        average_weights=[(1.0 - learning_rate) * t + learning_rate * e for t, e in zip(target_weights, evaluate_weights)]
        target_net.set_weights(average_weights)

    def decide(self,observation):
        probs=self.actor_net.predict(observation[np.newaxis])[0]
        action=np.random.choice(self.action_n, p=probs)
        return action

    def learn(self,observation,action,reward,next_observation,done):
        self.replayer.store(observation,action,reward,next_observation,done)

        if done:
            for batch in range(self.batches):
                observations, actions, rewards, next_observations, dones=self.replayer.sample(self.batch_size)
                pis=self.actor_net.predict(observations)
                q0s=self.q0_net.predict(observations)
                q1s=self.q1_net.predict(observations)

                self. actor_net.fit(observations, pis, verbose=0)
                q01s=np.minimum(q0s, q1s)
                entropic_q01s = q01s - self.alpha*np.log(pis)
                v_targets=self.v_target_net.predict(next_observations)
                self.v_evaluate_net.fit(observations, v_targets, verbose=0)

                next_vs=self.v_evaluate_net.predict(next_observations)
                q_targets=rewards + self.gamma * next_vs[:, 0] * (1 - dones)
                q0s[range(self.batch_size), actions]=q_targets
                q1s[range(self.batch_size), actions]= q_targets
                self.q0_net.fit(observations,q0s,verbose=0)
                self.q1_net.fit(observations,q1s,verbose=0)
                self.update_target_net(self.v_target_net, self.v_evaluate_net, self.net_learning_rate)


actor_kwargs={'hidden_sizes' : [100,],
              'learning_rate' : 0.01}
critic_kwargs={'hidden_sizes' : [100,],
               'learning_rate' : 0.01}
agent=SACAgent(env,actor_kwargs, critic_kwargs)

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

'''
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.savefig("8-6.png")
plt.show()
plt.close()
'''
#测试
agent.epsilon=0.0
episode_rewards=[play_qlearning(env,agent) for _ in range(100)]
print("平均回合奖励：{}/{} = {}".format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))

pd.DataFrame(agent.Q)
policy=np.eye(agent.action_n)[agent.Q.argmax(axis=-1)]
pd.DataFrame(policy)
