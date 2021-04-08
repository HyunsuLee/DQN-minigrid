import numpy as np
import torch
import torch.nn as nn

from gym_minigrid.wrappers import *
from drl_lib import dqn

GRID_ENV_NAME = 'MiniGrid-Empty-8x8-v0' 
#GRID_ENV_NAME = 'Asterix-v0'
env = gym.make(GRID_ENV_NAME)
env = RGBImgObsWrapper(env) 
env = ImgObsWrapper(env) #RGB space : 64 x 64 x 3
obs = env.reset()
print(env.observation_space.shape)
print(env.action_space.n)

net = dqn.DQN(env.observation_space.shape, env.action_space.n)
print(net)


for _ in range(2):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    obs = torch.tensor(obs)
    net(obs)
    #env.render()
    #print(obs['image'][:, :, 0])
#print(rew)
#print(done)
#print(info)

env.close()