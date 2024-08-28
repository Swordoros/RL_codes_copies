#SARSA agent和环境交互函数play SARSA

import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")

env=gym.make('Taxi-v3')

state=env.reset()
taxirow, taxicol, passloc, destidx = env.unwrapped.decode(state)


class SARSA_Agent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=0.01):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n=env.action_space.n
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self,state):
        if np.random.uniform() > self.epsilon:
            action=self.q[state].argmax()
        else:
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

