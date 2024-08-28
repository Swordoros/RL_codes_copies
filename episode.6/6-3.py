#独热编码（one-hot coding）改进的砖瓦编码（tile coding）
#针对连续性问题？


import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

env=gym.make('MountainCar-v0')

#定义瓦片编码类
class TileCoder:
    def __init__(self, layers, features):
        self.layers = layers
        self.features = features
        self.codebook = {}

    def get_code(self, codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count=len(self.codebook)
        if count>=self.features:
            return hash(codeword)%self.features
        else:
            self.codebook[codeword] = count
            return count

    def __call__(self, floats=(), ints=()):
        dim=len(floats)
        scaled_floats = [f*self.layers*self.layers for f in floats]
        features=[]
        for layer in range(self.layers):
            codeword=(layer,)+tuple(int((f+(1+dim*i)*layer)/self.layers) for i ,f in enumerate(scaled_floats)) + ints
            feature=self.get_code(codeword)
            features.append(feature)
        return features
