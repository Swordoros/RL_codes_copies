
import gym
import warnings
warnings.filterwarnings("ignore")

env = gym.make('CartPole-v0', render_mode='human')

from gym import envs

# 获取所有环境规格的 ID
env_ids = list(envs.registry.keys())
print(env_ids)

