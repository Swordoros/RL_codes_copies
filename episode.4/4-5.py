#带起始探索的同策略回合更新

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

def monte_carlo_with_exploring_start(env, episode_num=500000):
    policy=np.zeros((22,11,2,2))
    policy[:,:,:,1]=1.
    q=np.zeros_like(policy)
    c=np.zeros_like(policy)
    for _ in range(episode_num):
        state = (np.random.randint(12,22),
                 np.random.randint(1,11),
                 np.random.randint(2))
        action= np.random.randint(2)

        env.reset()
        if state[2]:  #有A
            env.player=[1, state[0]-11]
        else: #无A
            if state[0] == 21: #等于21
                env.player=[10, 9 ,2]
            else:  #小于21
                env.player=[10, state[0]-10]
        env.dealer[0] = state[1]
        state_actions=[]
        while True:
            state_actions.append((state,action))
            observation,reward,done, _=env.step(action)
            if done:
                break
            state=ob2state(observation)
            action=np.random.choice(env.action_space.n, p=policy[state])
        g=reward
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g-q[state][action]) / c[state][action]
            a=q[state].argmax()
            policy[state] = 0.
            policy[state][a] = 1.
        return policy, q

policy, q = monte_carlo_with_exploring_start(env)
v=q.max(axis=-1)
plot(policy.argmax(-1))
plot(v)