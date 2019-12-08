import numpy as np
import gym
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torch

from networks.CycleGan import Generator

N_CHANNELS = 1
SIZE = 84
USE_CUDA = torch.cuda.is_available()

netG_S2T = Generator(N_CHANNELS, N_CHANNELS)

netG_S2T.load_state_dict(torch.load("output/netG_Y2X.pkl"))

netG_S2T.eval()

Tensor = torch.cuda.FloatTensor if USE_CUDA else torch.Tensor
input_S = Tensor(1, N_CHANNELS, SIZE, SIZE)

if USE_CUDA:
    netG_S2T.cuda()


def translate_state(obs):
    #obs2 = np.array(obs)[None][0]
    obs2 = obs.transpose(2, 0, 1)
    for i in range(len(obs2)):
        #print(obs2[i].shape)
        #print(obs2[i])
        #plt.imshow(obs2[i], cmap="gray")
        #plt.show()
        real_S = Variable(input_S.copy_(torch.from_numpy(obs2[i])))
        fake_T = 0.5*(netG_S2T(real_S).data + 1.0)*255.0
        #print(torch.max(fake_T).item())
        obs2[i] = fake_T.cpu()
        #print(obs2[i].shape)
        #print(obs2[i])
        #save_image(real_S, 'output/S/%04d.png' % (i))
        #save_image(fake_T, 'output/T/%04d.png' % (i))
        #plt.imshow(obs2[i], cmap="gray")
        #plt.show()
    obs2 = obs2.transpose(1, 2, 0)
    return obs2