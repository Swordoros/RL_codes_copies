#策略评估的实现
import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def v2q(env, v, s=None, gamma=1.):
    if s is not None:
        q=np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                q[a] +=prob*(reward+gamma*v[next_state]*(1.-done))
    else:
        q=np.zeros((env.observation_space.n,env.action_space.n))
        for s in range(env.observation_space.n):
            q[s]=v2q(env,v,s,gamma)
    return q

def evaluate_policy(env, policy, gamma=1., tolerant=1e-6):
    v=np.zeros(env.observation_space.n)
    while True:
        delta=0
        for s in range(env.observation_space.n):
            vs=sum(policy[s] * v2q(env,v,s,gamma))
            delta=max(delta,np.abs(v[s]-vs))
            v[s]=vs
        if delta<tolerant:
            break
    return v