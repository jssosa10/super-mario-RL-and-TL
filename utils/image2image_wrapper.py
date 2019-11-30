import numpy as np
import gym
from torch.autograd import Variable
import torch

from networks.CycleGan import Generator

USE_CUDA = torch.cuda.is_available()

def _process_image2image(obs):
    obs = np.array(obs)[None][0]


class ProcessImage2Image(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessImage2Image, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame84(obs), reward, done, info

    def reset(self):
        return _process_frame84(self.env.reset())