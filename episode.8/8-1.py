# 动作价值A/C算法

import gym  # 导入gym库，用于创建和使用强化学习环境
import numpy as np  # 导入numpy库，用于数值计算
import warnings  # 导入warnings库，用于忽略警告
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import pandas as pd  # 导入pandas库，用于数据处理
import tensorflow as tf  # 导入tensorflow库，用于深度学习
import tensorflow.keras as keras  # 导入tensorflow的keras模块，用于构建神经网络
from tensorflow.keras.initializers import GlorotUniform  # 导入GlorotUniform初始化器
from tensorflow.keras.optimizers import Adam  # 导入Adam优化器

warnings.filterwarnings("ignore")  # 忽略警告

env = gym.make('Acrobot-v1')  # 创建Acrobot-v1环境

class QActorCriticAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs, gamma=0.99):
        self.action_n = env.action_space.n  # 获取动作空间的大小
        self.gamma = gamma  # 折扣因子
        self.discount = 1.  # 初始折扣值
        self.actor_net = self.build_network(output_size=self.action_n, output_activation=tf.nn.softmax, loss=keras.losses.categorical_crossentropy, **actor_kwargs)  # 构建actor网络
        self.critic_net = self.build_network(output_size=self.action_n, **critic_kwargs)  # 构建critic网络

    def build_network(self, hidden_sizes, output_size, input_size=None, activation=tf.nn.relu, output_activation=None, loss=keras.losses.mse, learning_rate=0.01):
        model = keras.Sequential()  # 创建一个顺序模型
        for idx, hidden_size in enumerate(hidden_sizes):
            kwargs = {}
            if idx == 0 and input_size is not None:
                kwargs['input_shape'] = (input_size,)  # 设置输入形状
            model.add(keras.layers.Dense(units=hidden_size, activation=activation, kernel_initializer=GlorotUniform(seed=0), **kwargs))  # 添加隐藏层
        model.add(keras.layers.Dense(units=output_size, activation=output_activation, kernel_initializer=GlorotUniform(seed=0)))  # 添加输出层
        optimizer = Adam(learning_rate)  # 创建Adam优化器，自适应调整学习率
        model.compile(optimizer=optimizer, loss=loss)  # 编译模型
        return model

    def decide(self, observation):
        probs = self.actor_net.predict(observation[np.newaxis])[0]  # 预测动作概率
        action = np.random.choice(self.action_n, p=probs)  # 根据概率选择动作
        return action

    def learn(self, observation, action, reward, next_observation, done, next_action=None):
        x = observation[np.newaxis]  # 将观察转换为批量形式
        u = self.critic_net.predict(x)  # 预测critic网络的输出
        q = u[0, action]  # 获取当前动作的Q值
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)  # 将输入转换为tensor
        with tf.GradientTape() as tape:
            pi_tensor = self.actor_net(x_tensor)[0, action]  # 获取动作的概率
            logpi_tensor = tf.math.log(tf.clip_by_value(pi_tensor, 1e-6, 1.0))  # 计算对数概率
            loss_tensor = -self.discount * q * logpi_tensor  # 计算损失
        grad_tensors = tape.gradient(loss_tensor, self.actor_net.variables)  # 计算梯度
        self.actor_net.optimizer.apply_gradients(zip(grad_tensors, self.actor_net.variables))  # 应用梯度

        # 训练critic网络
        u[0, action] = reward  # 更新Q值
        if not done:
            q = self.critic_net.predict(next_observation[np.newaxis])[0, 0]  # 预测下一个状态的Q值
            u[0, action] += self.gamma * q  # 更新Q值
        self.critic_net.fit(x, u, verbose=0)  # 训练critic网络

        if done:
            self.discount = 1.  # 重置折扣值
        else:
            self.discount *= self.gamma  # 更新折扣值

actor_kwargs = {'hidden_sizes': [100,], 'learning_rate': 0.0002}  # actor网络的超参数
critic_kwargs = {'hidden_sizes': [100,], 'learning_rate': 0.0005}  # critic网络的超参数

agent = QActorCriticAgent(env, actor_kwargs, critic_kwargs)  # 创建代理

def play_qlearning(env, agent, train=False, render=False):
    episode_reward = 0  # 初始化回合奖励
    observation = env.reset()  # 重置环境
    while True:
        if render:
            env.render()  # 渲染环境
        action = agent.decide(observation)  # 决定动作
        next_observation, reward, done, _ = env.step(action)  # 执行动作
        episode_reward += reward  # 累加奖励
        if train:
            agent.learn(observation, action, reward, next_observation, done, next_action=None)  # 学习
        if done:
            break  # 回合结束
        observation = next_observation  # 更新观察
    return episode_reward

episodes = 5000  # 总回合数
episode_rewards = []  # 存储每个回合的奖励
for i in range(episodes):
    episode_reward = play_qlearning(env, agent, train=True, render=True)  # 进行一个回合
    episode_rewards.append(episode_reward)  # 存储回合奖励
    if i % 100 == 0:
        print(f"Episode {i}: {episode_reward}")  # 打印进度

'''
plt.plot(episode_rewards)  # 绘制奖励曲线
plt.xlabel("Episode")  # x轴标签
plt.ylabel("Episode Reward")  # y轴标签
plt.savefig("8-1.png")  # 保存图像
plt.show()  # 显示图像
plt.close()  # 关闭图像
'''

# 测试
'''
agent.epsilon = 0.0  # 设置epsilon为0
episode_rewards = [play_qlearning(env, agent) for _ in range(100)]  # 进行100个测试回合
print("平均回合奖励：{}/{} = {}".format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))  # 打印平均奖励
'''
