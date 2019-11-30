import numpy as np
import gym
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch

from networks.CycleGan import Generator

N_CHANNELS = 1
SIZE = 84
USE_CUDA = torch.cuda.is_available()

netG_X2Y = Generator(N_CHANNELS, N_CHANNELS)

netG_X2Y.load_state_dict(torch.load("output/netG_X2Y.pkl"))

netG_X2Y.eval()

Tensor = torch.cuda.FloatTensor if USE_CUDA else torch.Tensor
input_X = Tensor(1, N_CHANNELS, SIZE, SIZE)

if USE_CUDA:
    netG_X2Y.cuda()


def _process_image2image(obs):
    obs2 = np.array(obs)[None][0]
    obs2 = obs2.transpose(2, 0, 1)
    print("TEST SHAPE:", torch.from_numpy(obs2[0]).shape)
    real_X = Variable(input_X.copy_(torch.from_numpy(obs2[0])))
    fake_Y = 0.5*(netG_X2Y(real_X).data + 1.0)
    plt.imshow(obs2[0], cmap="gray")
    plt.show()
    print("TEST shape output", fake_Y.cpu().numpy()[0][0].shape)
    plt.imshow(fake_Y.cpu().numpy()[0][0], cmap="gray")
    plt.show()
    return obs


class ProcessImage2Image(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessImage2Image, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_image2image(obs), reward, done, info

    def reset(self):
        return _process_image2image(self.env.reset())


def wrapImage(env):
    return ProcessImage2Image(env)
