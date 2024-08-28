#用bellman方程求解状态价值和运动价值


import gym
import warnings
warnings.filterwarnings("ignore")

env=gym.make('CliffWalking-v0')

import numpy as np
def play_once(env, policy):
    total_reward = 0
    state = env.reset()
    while True:
        loc=np.unravel_index(state,env.shape)
        print('状态={}，位置={}'.format(state,loc),end=' ')
        action=np.random.choice(env.nA, p=policy[state])
        state, reward, done, _ = env.step(action)
        print('动作={}，奖励={}'.format(action,reward))
        total_reward += reward
        if done:
            break
    return total_reward

'''
actions =np.ones(env.shape, dtype=int)
actions[-1,:]=0
actions[:,-1]=2
optimal_policy=np.eye(env.nA)[actions.reshape(-1)]
total_rewards=play_once(env, optimal_policy)
print('总奖励={}'.format(total_rewards))
'''
def evaluate_bellman(env, policy, gamma=1.):
    a,b=np.eye(env.nS), np.zeros(env.nS)
    for state in range(env.nS-1):
        for action in range(env.nA):
            pi=policy[state][action]
            for p, next_state, reward, done in env.P[state][action]:
                a[state,next_state] -=(pi*gamma*p)
                b[state] += (pi*reward*p)
    v=np.linalg.solve(a,b)
    q=np.zeros((env.nS,env.nA))
    for state in range(env.nS):
        for action in range(env.nS -1):
            for action in range(env.nA):
                for p, next_state, reward, done in env.P[state][action]:
                    q[state,action] += (p*(reward+gamma*v[next_state]))

    return v,q
