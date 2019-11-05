import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset

USE_CUDA = torch.cuda.is_available()
N_CHANNELS = 1
N_EPOCHS = 50
BATCH_SIZE = 10
LR = 0.0002
DECAY_EPOCH = 25
SIZE = 84
N_CPU = 12
EPOCH = 0

# Definition of variables ######
# Networks
netG_X2Y = Generator(N_CHANNELS, N_CHANNELS)
netG_Y2X = Generator(N_CHANNELS, N_CHANNELS)

if USE_CUDA:
    netG_X2Y.cuda()
    netG_Y2X.cuda()

# Load state dicts
netG_X2Y.load_state_dict(torch.load("output/netG_X2Y.pkl"))
netG_Y2X.load_state_dict(torch.load("output/netG_Y2X.pkl"))

# Set model's test mode
netG_X2Y.eval()
netG_Y2X.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if USE_CUDA else torch.Tensor
input_X = Tensor(BATCH_SIZE, N_CHANNELS, SIZE, SIZE)
input_Y = Tensor(BATCH_SIZE, N_CHANNELS, SIZE, SIZE)

# Dataset loader
transforms_ = [transforms.Grayscale(num_output_channels=1),
               transforms.ToTensor(),
               transforms.Normalize([0.5], [0.5])]

dataloader = DataLoader(ImageDataset("data", transforms_=transforms_, mode='test'),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=N_CPU)
###################################

# Testing######

# Create output dirs if they don't exist
if not os.path.exists('output/X'):
    os.makedirs('output/X')
if not os.path.exists('output/Y'):
    os.makedirs('output/Y')

for i, batch in enumerate(dataloader):
    # Set model input
    real_X = Variable(input_X.copy_(batch['X']))
    real_Y = Variable(input_Y.copy_(batch['Y']))

    # Generate output
    fake_Y = 0.5*(netG_X2Y(real_X).data + 1.0)
    fake_X = 0.5*(netG_Y2X(real_Y).data + 1.0)

    # Save image files
    save_image(fake_X, 'output/X/%04d.png' % (i+1))
    save_image(fake_Y, 'output/Y/%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
