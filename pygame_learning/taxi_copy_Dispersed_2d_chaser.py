from contextlib import closing
from io import StringIO
from os import path
from typing import Optional
import pygame
import sys
import time
import numpy as np

from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
from gym.utils.renderer import Renderer

#state= x,y,vy
#动作为调整y方向的速度
#显示范围为400*300


class RLPIDEnv(Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array", "single_rgb_array"],
                "render_fps": 30,
                }

    def __init__(self, render_mode: Optional[str] = None):
        self.targets = (300,200)
        self.x0=0
        self.y0=100
        self.vx0=2
        self.vy0=0
        self.vy_limit=[-2,-1,0,1,2]

        num_states = 12000*5   #12000个位置状态
        num_xs = 400
        num_ys = 300
        max_row = num_xs - 1
        max_col = num_ys - 1
        self.initial_state_distrib = (self.x0,self.y0,self.vy0)
        num_actions = 2
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }

        for x in range(num_xs):
            for y in range(num_ys):
                for vy in range(5):
                    for action in range(num_actions):

                        state = self.encode(x, y, vy)
                        new_vy=self.speed_shifter(vy,action)
                        new_x=x+2
                        new_y=y+vy
                        new_state = self.encode(new_x, new_y, new_vy)

                        if self.distance(new_x,new_y)<=10:
                            reward=100
                            terminated=True
                        else:
                            reward=-1
                            terminated=False
                        self.P[state][action].append((1.0, new_state, reward, terminated))  # 添加状态转移信息

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        self.window = None
        self.clock = None
        self.window_size=(400,300)
        self.taxi_imgs = None
        self.taxi_orientation = 0
        self.passenger_img = None
        self.destination_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None


    def speed_shifter(self, vy, action):
        if action == 0:
            vy=vy-1
            if vy<=-2:
                vy=-2
        if action == 1:
            vy=vy+1
            if vy>=2:
                vy=2
        return vy



    def encode(self, row, col, vy):
        # 40 rows, 30 columns, 5 states
        i = row
        i *= 30
        i += col
        i *= 5
        i += vy
        #print(i)
        return i

    def decode(self, i):
        out = []
        out.append(i % 5)  # state
        i = i // 5
        out.append(i % 30)  # col
        i = i // 30
        out.append(i)  # row
        assert 0 <= i < 40
        out = list(reversed(out))
        #print(out)
        return out

    def distance(self,x,y):
        return np.sqrt((x-self.targets[0])**2+(y-self.targets[1])**2)


    def step(self, a):
        transitions = self.P[self.s][a]  # 获取当前状态和动作对应的所有可能转移
        i = categorical_sample([t[0] for t in transitions], self.np_random)  # 根据概率分布随机选择一个转移
        p, s, r, t = transitions[i]  # 获取选择的转移的概率、新状态、奖励和是否终止
        self.s = s  # 更新当前状态
        self.lastaction = a  # 记录最后执行的动作
        self.renderer.render_step()  # 渲染当前步骤
        return (int(s), r, t, False,None)  # 返回新状态、奖励、是否终止、False和附加信息

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)  # 调用父类的reset方法，设置随机种子
        self.s = self.encode(self.x0, self.y0, self.vy0)  # 根据初始状态分布随机选择一个初始状态
        self.lastaction = None  # 重置最后执行的动作
        self.renderer.reset()  # 重置渲染器
        self.renderer.render_step()  # 渲染初始步骤
        if not return_info:  # 如果不需要返回额外信息
            return int(self.s)  # 返回初始状态
        else:  # 如果需要返回额外信息
            return int(self.s), {"prob": 1.0, "action_mask": None}  # 返回初始状态和附加信息

    def show(self):
        pygame.time.Clock().tick(30)
        # 初始化pygame
        pygame.init()

        # 设置屏幕大小
        screen_width, screen_height = 400, 300
        screen = pygame.display.set_mode((screen_width, screen_height))

        # 设置标题
        pygame.display.set_caption("2D_Chaser")
        point_x, point_y , point_vy = self.decode(self.s)

        running = True
        while running:
            point_x, point_y, point_vy = self.decode(self.s)
            # 更新位置
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    # 检查按键
                if event.type == pygame.KEYDOWN:
                    print(f"Key pressed: {pygame.key.name(event.key)}")
                    if event.key == pygame.K_LEFT:
                        sys.exit()

            # 填充背景色
            # 填充背景颜色
            screen.fill((255, 255, 255))
            ppx = []
            ppy = []
            # 绘制点
            pygame.draw.circle(screen, (255,0,0), (point_x, point_y), 5)
            # 留存点
            ppx.append(point_x)
            ppy.append(point_y)
            for i in range(len(ppx)):
                pygame.draw.circle(screen, (0,0,255), (ppx[i], ppy[i]), 3)
            #目标点
            pygame.draw.circle(screen, (0,0,0), (300,200), 10)
            # 更新屏幕显示
            pygame.display.flip()

            # 控制帧率
            pygame.time.Clock().tick(30)

        # 退出pygame
        pygame.quit()
        sys.exit()



