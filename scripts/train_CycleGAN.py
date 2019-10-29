import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.autograd import Variable
from PIL import Image
import torch

from networks.CycleGan import Generator
from networks.CycleGan import Discriminator
from utils.GAN_utils import ReplayBuffer
from utils.GAN_utils import LambdaLR
from utils.GAN_utils import weights_init_normal


USE_CUDA = torch.cuda.is_available()
N_CHANNELS = 1

# Define networks
netG_X2Y = Generator(N_CHANNELS, N_CHANNELS)
netG_Y2X = Generator(N_CHANNELS, N_CHANNELS)
netD_X = Discriminator(N_CHANNELS, N_CHANNELS)
netD_Y = Discriminator(N_CHANNELS, N_CHANNELS)

# use cuda
if USE_CUDA:
    netG_X2Y.cuda()
    netG_Y2X.cuda()
    netD_X.cuda()
    netD_Y.cuda()

netG_X2Y.apply(weights_init_normal)
netG_Y2X.apply(weights_init_normal)
netD_X.apply(weights_init_normal)
netD_Y.apply(weights_init_normal)

# Lossess
GAN_loss = torch.nn.MSELoss()
Cycle_loss = torch.nn.L1Loss()
identity_loss = torch.nn.L1Loss()
