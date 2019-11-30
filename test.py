from torch.utils.data import DataLoader
from utils.GAN_utils import ImageDataset
import torchvision.transforms as transforms
import torch

USE_CUDA = torch.cuda.is_available()
N_CHANNELS = 1
N_EPOCHS = 50
BATCH_SIZE = 1
LR = 0.0002
DECAY_EPOCH = 25
SIZE = 84
N_CPU = 2
EPOCH = 0

transforms_ = [transforms.Grayscale(num_output_channels=1),
               transforms.ToTensor()]

dataloader = DataLoader(ImageDataset("data", transforms_=transforms_, mode='test'),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=N_CPU)

for i, batch in enumerate(dataloader):
    print(batch['X'].shape)
