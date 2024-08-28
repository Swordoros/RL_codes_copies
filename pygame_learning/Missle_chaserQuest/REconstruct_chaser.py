import pygame
import sys
import time
import numpy as np
import math
from gym.envs.toy_text.utils import categorical_sample
import pickle

#state= x,y,vy
#动作为调整y方向的速度
#显示范围为(200*150)*4


def info():
    Screen_size = (200, 150)
    target_pos = (180, 120)
    num_actions = 2
    num_xs , num_ys=Screen_size
    num_vys = 5
    num_states = num_xs * num_ys * num_vys
    return  num_xs, num_ys, num_vys, num_states, num_actions, Screen_size, target_pos

def encode(row, col, state):
    num_xs, num_ys, num_vys, num_states, num_actions, _, _ = info()
    # 200 rows, 150 columns, 5 states
    i = row
    i *= num_ys
    i += col
    i *= num_vys
    i += state
    #print(i)
    return int(i)

def decode(i):
    out = []
    num_xs, num_ys, num_vys, num_states, num_actions, _, _ = info()
    out.append(i % 5)  # state
    i = i // num_vys
    out.append(i % 300)  # col
    i = i // num_ys
    out.append(i)  # row
    assert 0 <= i <= num_xs
    out = list(reversed(out))
    #print(out)
    return out


def speed_shifter(vy_code, action):
    if action == 0:
        vy_code=vy_code-1
        if vy_code<=0:
            vy_code=0
    if action == 1:
        vy_code=vy_code+1
        if vy_code>=4:
            vy_code=4
    return vy_code

def distance(x,y,targets):
    dis=np.sqrt((x-targets[0])**2+(y-targets[1])**2)
    result=math.floor(dis)
    return result


def P_init():
    num_xs, num_ys, num_vys, num_states, num_actions, screen_size, target_pos = info()
    screen_width, screen_height = screen_size
    x0=0
    y0=num_ys/2
    vx0=2
    vy0=2
    vy_list=[-2,-1,0,1,2]

    initial_state=encode(x0,y0,vy0)

    P = {
        state: {action: [] for action in range(num_actions)}
        for state in range(num_states)
    }
    for x in range(num_xs):
        for y in range(num_ys):
            for vy_code in range(5):
                for action in range(num_actions):

                    state = encode(x, y, vy_code)
                    #print(x, y, vy, action)
                    #print(state)
                    new_vy = speed_shifter(vy_code, action)
                    new_x = x + 2
                    new_y = y + vy_list[vy_code]
                    if new_y < 0 or new_y >= num_ys:
                        new_y = y
                    if new_x < 0 or new_x >= num_xs:
                        new_x = x
                    new_state = encode(new_x, new_y, new_vy)
                    if distance(new_x, new_y, target_pos) <= 30:
                        reward = 200
                        terminated = True
                    elif distance(new_x, new_y, target_pos) <= 20:
                        reward = 300
                        terminated = True
                    else:
                        reward = -1*distance(x, y, target_pos)/distance(x0, y0, target_pos)
                        reward=round(reward, 2)
                        if new_x < 0 or new_x >= screen_width-2 or new_y <=1  or new_y >= screen_height-2:
                            terminated = True
                        else:
                            terminated = False
                    P[state][action].append((1.0, new_state, reward, terminated))  # 添加状态转移信息
    return P


def state_init():
    num_xs, num_ys, num_vys, num_states, num_actions, _, _ = info()
    x0=0
    y0=num_ys/2
    vy0=2
    state0=encode(x0,y0,vy0)
    return state0


def step(P,state, action):
    transitions = P[state][action]
    i = categorical_sample([t[0] for t in transitions], np.random.RandomState())  # 根据概率分布随机选择一个转移
    p, s, r, t = transitions[i]
    return  (int(s), r, t, False , None)




def save_P_to_file(P, filename='P_data.txt'):
    with open(filename, 'wb') as file:
        pickle.dump(P, file)

def load_P_from_file(filename='P_data.txt'):
    with open(filename, 'rb') as file:
        P = pickle.load(file)
    return P
'''
def show(P, action):
    num_xs, num_ys, num_vys, num_states, num_actions, Screen_size, target_pos = info()
    p=P
    state=state_init()
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
        screen.fill((255, 255, 255))
        pygame.draw.circle(screen, (255, 0, 0), (point_x*4, point_y*4), 5)
        action=np.random.randint(2)  # 有策略了记得删除
        state, r , t , _, _=step(p,state,action)
        reward += r
        tgt_x, tgt_y = target_pos
        pygame.draw.circle(screen, (0, 0, 255), (tgt_x*4, tgt_y*4), 5)
        pygame.display.flip()
        pygame.time.Clock().tick(10)'''

# 更新环境时再重新运行下列代码
#P=P_init()
#save_P_to_file(P)

#P=load_P_from_file()
#print(P)

#show(P, None)


