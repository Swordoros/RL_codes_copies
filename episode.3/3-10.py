#利用价值迭代算法求解最优策略

import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")

env=gym.make('FrozenLake-v1')
env=env.unwrapped

def play_policy(env, policy,render=False):
    total_reward=0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action=np.random.choice(env.action_space.n, p=policy[observation])
        observation, reward, done, _, info= env.step(action)
        total_reward+=reward

        if done:
            break

    return total_reward

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

def improve_policy(env, v, policy, gamma=1.):
    optimal=True
    for s in range(env.observation_space.n):
        q=v2q(env,v,s,gamma)
        a=np.argmax(q)
        if policy[s][a] != 1.:
            optimal=False
            policy[s]=0
            policy[s][a]=1.
    return optimal


random_policy=np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
v_random = evaluate_policy(env, random_policy)
policy=random_policy.copy()
optimal=improve_policy(env, v_random, policy)

def iterate_policy(env, gamma=1., tolerant=1e-6):
    policy=np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    while True:
        v=evaluate_policy(env, policy, gamma, tolerant)
        if improve_policy(env, v, policy):
            break
    return policy, v

def iterate_value(env, gamma=1., tolerant=1e-6):
    v=np.zeros(env.observation_space.n)
    while True:
        delta=0
        for s in range(env.observation_space.n):
            vmax=np.max(v2q(env,v,s,gamma))
            delta=max(delta,np.abs(v[s]-vmax))
            v[s]=vmax
        if delta<tolerant:
            break
    policy=np.zeros((env.observation_space.n, env.action_space.n))
    for s in range(env.observation_space.n):
        a=np.argmax(v2q(env,v,s,gamma))
        policy[s][a]=1.
    return policy, v

policy_vi, v_vi=iterate_value(env)
print('状态价值函数=')
print(v_vi.reshape(4,4))
print('最优策略=')
print(np.argmax(policy, axis=1).reshape(4,4))

N=1000
episode_rewards=[play_policy(env, policy_vi) for _ in range(N)]
print('价值迭代 {}次平均奖励= {}'.format(N, np.mean(episode_rewards)))