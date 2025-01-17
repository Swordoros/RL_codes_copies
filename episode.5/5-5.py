#SARSA agentc测试和可视化展示

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
            #触发贪心策略了就直接选该状态最大值
            action=self.Q[state].argmax()
        else:
            #没触发贪心策略就随机选一个动作
            action=np.random.randint(0, self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done,next_action):
        u=reward+self.gamma*self.Q[next_state][next_action]*(1.0-done)
        td_error=u-self.Q[state][action]
        self.Q[state][action]+=self.learning_rate*td_error

agent=SARSA_Agent(env)

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
'''
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.savefig("5-5.png")
plt.show()
plt.close()
'''
#测试
agent.epsilon=0.0 #测试时不探索
#是否可视化展示
render=0
episode_rewards=[play_SARSA(env, agent, render=render) for _ in range(100)]
print("平均回合奖励：{}/{} = {}".format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))

pd.DataFrame(agent.Q)
policy=np.eye(agent.action_n)[agent.Q.argmax(axis=-1)]
pd.DataFrame(policy)