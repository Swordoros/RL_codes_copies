#重要性采样策略评估
#异策略策略评估

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

def evaluate_monte_carlo_importance_resample(env, policy, behavior_policy, episode_num=500000):
    q=np.zeros_like(policy)
    c=np.zeros_like(policy)
    for _ in range(episode_num):
        state_actions=[]
        observation = env.reset()
        while True:
            state=ob2state(observation)
            action=np.random.choice(env.action_space.n, p=behavior_policy[state])
            state_actions.append((state,action))
            observation, reward, done, _=env.step(action)
            if done:
                break
        g=reward
        rho=1. #重要性采样比率
        for state,action in reversed(state_actions):
            c[state][action]+=rho
            q[state][action]+=(rho/c[state][action]*(g-q[state][action]))
            rho*=(policy[state][action]/behavior_policy[state][action])
            if rho ==0:
                break
    return q

policy=np.zeros((22,11,2,2))
policy[20:,:,:,0] = 1
policy[:20,:,:,1] = 1
behavior_policy=np.ones_like(policy) * 0.5
q=evaluate_monte_carlo_importance_resample(env, policy, behavior_policy)
v=(q*policy).sum(axis=-1)
plot(v)