#最优动作价值确定最有确定性策略

#用线性规划求解bellman最优方程

import scipy
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
        #print('状态={}，位置={}'.format(state,loc),end=' ')
        action=np.random.choice(env.nA, p=policy[state])
        state, reward, done, _ = env.step(action)
        #print('动作={}，奖励={}'.format(action,reward))
        total_reward += reward
        if done:
            break
    return total_reward


actions =np.ones(env.shape, dtype=int)
actions[-1,:]=0
actions[:,-1]=2
optimal_policy=np.eye(env.nA)[actions.reshape(-1)]
total_rewards=play_once(env, optimal_policy)
#print('总奖励={}'.format(total_rewards))

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

policy=np.random.uniform(size=(env.nS,env.nA))
policy=policy/np.sum(policy,axis=1)[:, np.newaxis]
state_values, action_values=evaluate_bellman(env, policy)
optimal_state_values, optimal_action_values=evaluate_bellman(env, optimal_policy)

def optimal_bellman(env, gamma=1.):
    p=np.zeros((env.nS,env.nA,env.nS))
    r=np.zeros((env.nS,env.nA))
    for state in range(env.nS-1):
        for action in range(env.nA):
            for prob, next_state, reward, done in env.P[state][action]:
                p[state,action,next_state] += prob
                r[state,action] += prob*reward
    c=np.ones(env.nS)
    a_ub=gamma * p.reshape(-1,env.nS) - np.repeat(np.eye(env.nS),env.nA,axis=0)
    b_ub=-r.reshape(-1)
    a_eq=np.zeros((0,env.nS))
    b_eq=np.zeros(0)
    bounds=[(None,None),]*env.nS
    res=scipy.optimize.linprog(c,a_ub,b_ub,a_eq,b_eq,bounds=bounds,method='interior-point')
    v=res.x
    q=r+gamma*np.dot(p,v)
    return v,q

optimal_state_values, optimal_action_values=optimal_bellman(env)
optimal_actions=optimal_action_values.argmax(axis=1)
print("最优策略= {}".format(optimal_actions))
print(optimal_actions[:12])
print(optimal_actions[13:25])
print(optimal_actions[25:37])
print(optimal_actions[37:],'[终点]')