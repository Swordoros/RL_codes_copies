#基于柔性策略的同策略回合更新

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

env=gym.make('Blackjack-v1')
observation=env.reset()

def ob2state(observation):
    return (observation[0],observation[1],int(observation[2]))


def plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    titles=['without ace', 'with ace']
    have_ace=[0, 1]
    extent=[12, 22, 1, 11]
    for title, have_ace, axis in zip(titles, have_ace, axes):
        dat=data[extent[0]:extent[1],extent[2]:extent[3],have_ace].T
        axis.imshow(dat,extent=extent,origin='lower')
        axis.set_ylabel('Dealer showing')
        axis.set_xlabel('Player sum')
        axis.set_title(title)
    plt.show()

def monte_carlo_with_soft(env, episodes_sum=500000, epsilon=0.1):
    policy=np.ones((22, 11, 2, 2)) * 0.5
    q=np.zeros_like(policy)
    c=np.zeros_like(policy)
    for _ in range(episodes_sum):
        state_actions=[]
        observation=env.reset()
        while True:
            state=ob2state(observation)
            action=np.random.choice(env.action_space.n,p=policy[state])
            state_actions.append((state,action))
            observation,reward,done, _=env.step(action)
            if done:
                break
        g=reward
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g-q[state][action])/c[state][action]

            #更新策略为柔性策略
            a=q[state].argmax()
            policy[state]=epsilon/2.
            policy[state][a] += (1.-epsilon)
    return policy, q

policy, q=monte_carlo_with_soft(env)
v=q.max(axis=-1)
plot(policy.argmax(-1))
plot(v)

