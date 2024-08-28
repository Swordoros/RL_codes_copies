#同策略回合更新策略评估

import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")

env=gym.make('Blackjack-v1')

def ob2state(observation):
    return (observation[0],observation[1],int(observation[2]))

def evaluate_action_monte_carlo(env, policy, episodes_sum=500000):
    # 初始化动作价值函数 q，形状与策略相同，初始值为0
    q = np.zeros_like(policy)

    # 初始化计数器 c，形状与策略相同，初始值为0
    c = np.zeros_like(policy)

    # 进行 episodes_sum 次回合模拟
    for i in range(episodes_sum):
        # 初始化状态-动作对列表
        state_actions = []

        # 重置环境，获取初始观测
        observation = env.reset()

        # 进入一个无限循环，直到回合结束
        while True:
            # 将观测转换为状态
            state = ob2state(observation)

            # 根据策略选择动作
            action = np.random.choice(env.action_space.n, p=policy[state])

            # 记录状态和动作
            state_actions.append((state, action))

            # 执行动作，获取新的观测、奖励、是否结束标志和其他信息
            observation, reward, done, _ = env.step(action)

            # 如果回合结束，跳出循环
            if done:
                break

        # 记录最终的奖励
        g = reward

        # 遍历所有记录的状态-动作对
        for state, action in state_actions:
            # 更新计数器
            c[state][action] += 1.

            # 更新动作价值函数，使用增量式更新方法
            q[state][action] += (g - q[state][action]) / c[state][action]

    # 返回动作价值函数
    return q


# 初始化策略，大小为 (22, 11, 2, 2)，初始值为0
policy = np.zeros((22, 11, 2, 2))

# 规定大于20点时玩家选择stand，即将策略中大于等于20点的部分设置为1
policy[20:, :, :, 0] = 1

# 规定小于20点时玩家选择hit，即将策略中小于20点的部分设置为1
policy[:20, :, :, 1] = 1

# 使用蒙特卡洛方法评估策略，获取动作价值函数
q = evaluate_action_monte_carlo(env, policy)

# 计算状态价值函数，通过将动作价值函数与策略相乘并求和得到
v = (q * policy).sum(axis=-1)

print(q)