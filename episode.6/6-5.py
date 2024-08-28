#函数近似SARSA算法agent

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

env = gym.make('MountainCar-v0')


# 定义瓦片编码类
class TileCoder:
    def __init__(self, layers, features):
        self.layers = layers
        self.features = features
        self.codebook = {}

    def get_code(self, codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count >= self.features:
            return hash(codeword) % self.features
        else:
            self.codebook[codeword] = count
            return count

    def __call__(self, floats=(), ints=()):
        dim = len(floats)
        scaled_floats = [f * self.layers * self.layers for f in floats]
        features = []
        for layer in range(self.layers):
            codeword = (layer,) + tuple(
                int((f + (1 + dim * i) * layer) / self.layers) for i, f in enumerate(scaled_floats)) + ints
            feature = self.get_code(codeword)
            features.append(feature)
        return features


class SARSA_Agent:
    def __init__(self, env, layers=8, features=1893, gamma=1.0, learning_rate=0.03, epsilon=0.001):
        self.action_n = env.action_space.n
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        self.obs_scale = env.observation_space.high-env.observation_space.low
        self.encoder = TileCoder(layers, features)
        self.w=np.zeros(features)
        self.gamma=gamma
        self.learning_rate=learning_rate
        self.epsilon=epsilon

    def encode(self,observation,action):
        state=tuple((observation-self.obs_low)/self.obs_scale)
        actions=(action,)
        return self.encoder(state,actions)

    def get_q(self,observation,action):
        features=self.encode(observation,action)
        return self.w[features].sum()

    def decide(self,observation):
        if np.random.uniform(0,1)<self.epsilon:
            return np.random.choice(self.action_n)
        else:
            qs=[self.get_q(observation,a) for a in range(self.action_n)]
            return np.argmax(qs)

    def learn(self,observation,action,reward,next_observation,done,next_action):
        u=reward+ (1. - done)*self.gamma*self.get_q(next_observation,next_action)
        td_error=u-self.get_q(observation,action)
        features=self.encode(observation,action)
        self.w[features]+=(self.learning_rate*td_error)

class SARSA_lamda_Agent(SARSA_Agent):
    def __init__(self, env, layers=8, features=1893, gamma=1.0, learning_rate=0.03, epsilon=0.001, lamda=0.9):
        super().__init__(env=env, layers=layers, features=features, gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
        self.lamda=lamda
        self.z=np.zeros(features) #初始化资格迹

    def learn(self,observation,action,reward,next_observation,done,next_action):
        u=reward
        if not done:
            u=u+self.gamma*self.get_q(next_observation,next_action)
            self.z *=(self.gamma*self.lamda)
            features=self.encode(observation,action)
            self.z[features] = 1.0
        td_error=u-self.get_q(observation,action)
        self.w+=(self.learning_rate*td_error*self.z)
        if done:
            self.z=np.zeros_like(self.z) #清空资格迹


agent=SARSA_lamda_Agent(env)



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



#开始训练
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
plt.savefig("6-5.png")
plt.show()
plt.close()

agent.epsilon=0.0 #测试时不探索
#是否可视化展示
render=True
episode_rewards=[play_SARSA(env, agent, render) for _ in range(10)]
print("平均回合奖励：{}/{} = {}".format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))
