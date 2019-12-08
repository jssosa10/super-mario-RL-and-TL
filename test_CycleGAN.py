import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt


from networks.CycleGan import Generator
from utils.GAN_utils import ImageDataset

USE_CUDA = torch.cuda.is_available()
N_CHANNELS = 1
N_EPOCHS = 50
BATCH_SIZE = 1
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
               transforms.ToTensor()]

dataloader = DataLoader(ImageDataset("data", transforms_=transforms_, mode='test'),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=N_CPU)
###################################

# Testing######

# Create output dirs if they don't exist
if not os.path.exists('output/X'):
    os.makedirs('output/X')
if not os.path.exists('output/Y'):
    os.makedirs('output/Y')
if not os.path.exists('output/X_REAL'):
    os.makedirs('output/X_REAL')
if not os.path.exists('output/Y_REAL'):
    os.makedirs('output/Y_REAL')

for i, batch in enumerate(dataloader):
    # Set model input
    real_X = Variable(input_X.copy_(batch['X']))
    real_Y = Variable(input_Y.copy_(batch['Y']))

    # Generate output
    fake_Y = 0.5*(netG_X2Y(real_X).data + 1.0)
    fake_X = 0.5*(netG_Y2X(real_Y).data + 1.0)

    plt.imshow(real_Y.cpu().numpy()[0][0], cmap="gray")
    plt.show()

    plt.imshow(fake_X.cpu().numpy()[0][0], cmap="gray")
    plt.show()

    # Save image files
    #save_image(fake_X, 'output/X/%04d.png' % (i+1))
    #save_image(real_X, 'output/X_REAL/%04d.png' % (i+1))
    #save_image(fake_Y, 'output/Y/%04d.png' % (i+1))
    #save_image(real_Y, 'output/Y_REAL/%04d.png' % (i+1))

    #sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
