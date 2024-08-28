import pygame
import sys
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from REconstruct_chaser import info, step, encode , decode, load_P_from_file

def info2():
    gamma = 0.99
    learning_rate= 0.1
    epsilon = 0.01
    return gamma, learning_rate, epsilon

def Q_init():
    num_xs, num_ys, num_vys, num_states, num_actions, _, _ = info()
    Q = np.zeros((num_states, num_actions))
    return Q

def save_Q_to_file(Q, filename='Q_data.txt'):
    with open(filename, 'wb') as file:
        pickle.dump(Q, file)

def load_Q_from_file(filename='Q_data.txt'):
    with open(filename, 'rb') as file:
        Q = pickle.load(file)
    return Q


#Q=Q_init()
#save_Q_to_file(Q)

def learn(Q, state, action , reward, next_state, done , next_action):
    gamma, learning_rate, epsilon=info2()
    u = reward + gamma * Q[next_state][next_action] * (1.0 - done)
    td_error = u - Q[state][action]
    Q[state][action] += learning_rate * td_error
    return Q


def decide(Q, state, epsilon=0.01):
    num_xs, num_ys, num_vys, num_states, num_actions, _, _ = info()
    gamma, learning_rate, epsilon=info2()
    if np.random.uniform() > epsilon:
        # 触发贪心策略了就直接选该状态最大值
        action = Q[state].argmax()
    else:
        # 没触发贪心策略就随机选一个动作
        action = np.random.randint(0, num_actions)
    return action

def train_SARSA(Q,P, train=False):
    num_xs, num_ys, num_vys, num_states, num_actions, _, _ = info()
    episode_reward = 0
    state = encode(0, num_ys/2, 2)
    action=decide(Q, state)
    while True:
        next_state, reward, done, _, _ = step(P, state, action)
        #print(next_state, reward)
        decode(next_state)
        episode_reward += reward
        next_action=decide(Q, state)
        if train:
            Q=learn(Q, state, action, reward, next_state, done, next_action)
        if done:
            break
        action=next_action
        state=next_state
    return round(episode_reward,2)


Q=load_Q_from_file()
P=load_P_from_file()

episodes=100000
episode_rewards=[]
Average_rewards=[]
for i in range(episodes):
    episode_reward=train_SARSA(Q, P, train=True)
    episode_rewards.append(episode_reward)
    print('当前轮次:{}'.format(i+1))
    print(f"Episode {i+1}: {episode_reward}")
    avg=np.mean(episode_rewards[-1000:])
    Average_rewards.append(avg)
    print('近1000次平均奖励:{}'.format(round(avg,3)))
    if avg>=140 and i>3000 :
        print('===========================================')
        print('平均奖励达到140，停止训练')
        print('===========================================')
        break
print('自动存储中...')
save_Q_to_file(Q)
print('自动存储成功')

plt.plot(episode_rewards)
plt.plot(Average_rewards)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('SARSA Agent')
plt.savefig("rewards.png")
plt.show()
plt.close()
'''
Q=load_Q_from_file()
def show(P):
    num_xs, num_ys, num_vys, num_states, num_actions, Screen_size, target_pos = info()
    p=P
    num_xs, num_ys, num_vys, num_states, num_actions, _, _ = info()
    x0 = 0
    y0 = num_ys / 2
    vy0 = 2
    state = encode(x0, y0, vy0)
    screen_width, screen_height = Screen_size
    screen = pygame.display.set_mode((screen_width*4, screen_height*4))
    pygame.display.set_caption("Running Spot")
    running = True
    reward=0
    while running:
        x0, y0, vy0 = decode(state)
        print('当前总奖励为： {}'.format(round(reward, 2)))
        point_x = x0
        point_y = y0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                # 检查按键
            if event.type == pygame.KEYDOWN:
                print(f"Key pressed: {pygame.key.name(event.key)}")
                if event.key == pygame.K_LEFT:
                    sys.exit()
        if point_x < 0 or point_x >= screen_width-2 or point_y <=1  or point_y >= screen_height-2:
            sys.exit()
        if np.sqrt((point_x-target_pos[0])**2+(point_y-target_pos[1])**2)<=30:
            sys.exit()
        screen.fill((255, 255, 255))
        pygame.draw.circle(screen, (255, 0, 0), (point_x*4, point_y*4), 5)
        action=decide(Q, state, epsilon=0.0)
        state, r , t , _, _=step(p,state,action)
        reward += r
        tgt_x, tgt_y = target_pos
        pygame.draw.circle(screen, (0, 0, 255), (tgt_x*4, tgt_y*4), 10)
        pygame.display.flip()
        pygame.time.Clock().tick(10)

show(P)
'''