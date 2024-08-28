#柔性策略重要性采样最优策略求解
#异策略最优策略求解

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

env=gym.make('Blackjack-v1')


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

def monte_carlo_imprtance_resample(env, episode_num=500000):
    policy=np.zeros((22,11,2,2))
    policy[:,:,:,0]=1.
    behavior_policy=np.ones_like(policy)*0.5 #柔性策略
    q=np.zeros_like(policy)
    c=np.zeros_like(policy)
    for _ in range(episode_num):
        state_action=[]
        observation=env.reset()
        while True:
            state=ob2state(observation)
            action=np.random.choice(env.action_space.n,p=behavior_policy[state])
            state_action.append((state,action))
            observation,reward,done, _=env.step(action)
            if done:
                break
        g=reward
        rho=1.
        for state,action in reversed(state_action):
            c[state][action] += rho
            q[state][action] += (rho / c[state][action] * (g-q[state][action]))
            #策略改进
            a=q[state].argmax()
            policy[state][a]=1.
            policy[state]=0.
            if a != action:
                break
            rho /= behavior_policy[state][action]
    return policy,q

policy,q=monte_carlo_imprtance_resample(env)
v=q.max(axis=-1)

plot(policy.argmax(-1))
plot(v)