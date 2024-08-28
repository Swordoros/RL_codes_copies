#绘制最后一维的指标为0或1的3维数组

import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

env=gym.make('Blackjack-v1')
observation=env.reset()

def ob2state(observation):
    return (observation[0],observation[1],int(observation[2]))

def evaluate_action_monte_carlo(env, policy, episodes_sum=500000):
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
    return q

policy=np.zeros((22,11,2,2))
policy[20:,:,:,0]=1  #规定大于20点时玩家选择stand
policy[:20,:,:,1]=1  #规定小于20点时玩家选择hit
q=evaluate_action_monte_carlo(env,policy)
v=(q*policy).sum(axis=-1)

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

plot(v)
#策略评估得到的状态函数图像


