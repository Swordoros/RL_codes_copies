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

#四个动作：加加速度 减加速度 加左向角速度 加右向角速度
#state=(x,y,V,theta,round)  %x,y为坐标，V为速度，dtheta为角速度，round为当前的round数,pygame每步行动增加一个round
#奖励设置： 每round：时间惩罚-1，当前xy和目标xy的欧氏距离/初始距离*奖励系数rr， 到达目标欧氏距离10范围内奖励100

class RLPIDEnv(Env):
    metadata = {"render.modes": ["human", "rgb_array"],
                'WINDOW_SIZE' : (1200, 900),
                'FPS' : 60,
                }

    def __init__(self, render_mode: Optional[str] = None):
        self.targets = (1000,600)
        self.respawn = (0,600)
        self.v0=0
        self.dV0=0
        self.theta0=0
        self.dtheta0=0
        self.dV_range = (-1,3)
        self.dtheta_range=(-0.05,0.05)

    def rewarding(self,state):
        x,y,V,theta,round = state
        rr=0.8
        time_punishment=-1*round
        distance=((self.targets[0]-x)**2+(self.targets[1]-y)**2)**0.5
        reward=rr*distance+time_punishment
        if distance<10:
            reward=100
        return reward