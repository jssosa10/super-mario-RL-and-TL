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
from utils.GAN_utils import ImageDataset
from torch.utils.tensorboard import SummaryWriter

# Define HyperParameters
USE_CUDA = torch.cuda.is_available()
N_CHANNELS = 1
N_EPOCHS = 50
BATCH_SIZE = 10
LR = 0.0002
DECAY_EPOCH = 25
SIZE = 84
N_CPU = 12
EPOCH = 0

# Define networks
netG_X2Y = Generator(N_CHANNELS, N_CHANNELS)
netG_Y2X = Generator(N_CHANNELS, N_CHANNELS)
netD_X = Discriminator(N_CHANNELS)
netD_Y = Discriminator(N_CHANNELS)

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


# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_X2Y.parameters(), netG_Y2X.parameters()),
                               lr=LR, betas=(0.5, 0.999))
optimizer_D_X = torch.optim.Adam(netD_X.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D_Y = torch.optim.Adam(netD_Y.parameters(), lr=LR, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(N_EPOCHS, EPOCH, DECAY_EPOCH).step)
lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda=LambdaLR(N_EPOCHS, EPOCH, DECAY_EPOCH).step)
lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda=LambdaLR(N_EPOCHS, EPOCH, DECAY_EPOCH).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if USE_CUDA else torch.Tensor
input_X = Tensor(BATCH_SIZE, N_CHANNELS, SIZE, SIZE)
input_Y = Tensor(BATCH_SIZE, N_CHANNELS, SIZE, SIZE)
target_real = Variable(Tensor(BATCH_SIZE).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(BATCH_SIZE).fill_(0.0), requires_grad=False)

fake_X_buffer = ReplayBuffer()
fake_Y_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [transforms.Grayscale(num_output_channels = 1),
               transforms.Resize(int(SIZE*1.12), Image.BICUBIC),
               transforms.RandomCrop(SIZE),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize([0.5], [0.5])]

dataloader = DataLoader(ImageDataset("data", transforms_=transforms_, unaligned=True),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CPU)

# logger
writer = SummaryWriter()

# Training ######
for epoch in range(EPOCH, N_EPOCHS):
    print(epoch)
    for i, batch in enumerate(dataloader):
        print(i)
        # Set model input
        real_X = Variable(input_X.copy_(batch['X']))
        real_Y = Variable(input_Y.copy_(batch['Y']))

        # Generators X2Y and Y2X ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_X2Y(Y) should equal Y if real Y is fed
        same_Y = netG_X2Y(real_Y)
        loss_identity_Y = identity_loss(same_Y, real_Y)*5.0

        # G_Y2X(X) should equal X if real X is fed
        same_X = netG_Y2X(real_X)
        loss_identity_X = identity_loss(same_X, real_X)*5.0

        # GAN loss
        fake_Y = netG_X2Y(real_X)
        pred_fake = netD_Y(fake_Y)
        loss_GAN_X2Y = GAN_loss(pred_fake, target_real)

        fake_X = netG_Y2X(real_Y)
        pred_fake = netD_X(fake_X)
        loss_GAN_Y2X = GAN_loss(pred_fake, target_real)

        # Cycle loss
        recovered_X = netG_Y2X(fake_Y)
        loss_cycle_XYX = Cycle_loss(recovered_X, real_X)*10.0

        recovered_Y = netG_X2Y(fake_X)
        loss_cycle_YXY = Cycle_loss(recovered_Y, real_Y)*10.0

        # Total loss
        loss_G = loss_identity_X + loss_identity_Y + loss_GAN_X2Y + loss_GAN_Y2X + loss_cycle_XYX + loss_cycle_YXY
        loss_G.backward()

        optimizer_G.step()
        ###################################

        # Discriminator X ######
        optimizer_D_X.zero_grad()

        # Real loss
        pred_real = netD_X(real_X)
        loss_D_real = GAN_loss(pred_real, target_real)

        # Fake loss
        fake_X = fake_X_buffer.push_and_pop(fake_X)
        pred_fake = netD_X(fake_X.detach())
        loss_D_fake = GAN_loss(pred_fake, target_fake)

        # Total loss
        loss_D_X = (loss_D_real + loss_D_fake)*0.5
        loss_D_X.backward()

        optimizer_D_X.step()
        ###################################

        # Discriminator Y ######
        optimizer_D_Y.zero_grad()

        # Real loss
        pred_real = netD_Y(real_Y)
        loss_D_real = GAN_loss(pred_real, target_real)

        # Fake loss
        fake_Y = fake_Y_buffer.push_and_pop(fake_Y)
        pred_fake = netD_Y(fake_Y.detach())
        loss_D_fake = GAN_loss(pred_fake, target_fake)

        # Total loss
        loss_D_Y = (loss_D_real + loss_D_fake)*0.5
        loss_D_Y.backward()

        optimizer_D_Y.step()
        ###################################

        writer.add_scalar('Loss_G', loss_G, epoch)
        writer.add_scalar('Loss_G_identity', loss_identity_X + loss_identity_Y, epoch)
        writer.add_scalar('Loss_G_GAN', loss_GAN_X2Y + loss_GAN_Y2X, epoch)
        writer.add_scalar('Loss_G_cycle', (loss_cycle_XYX + loss_cycle_YXY), epoch)
        writer.add_scalar('Loss_D', loss_D_X + loss_D_Y, epoch)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_X.step()
    lr_scheduler_D_Y.step()

    # Save models checkpoints
    torch.save(netG_X2Y.state_dict(), 'output/netG_X2Y.pkl')
    torch.save(netG_Y2X.state_dict(), 'output/netG_Y2X.pkl')
    torch.save(netD_X.state_dict(), 'output/netD_X.pkl')
    torch.save(netD_Y.state_dict(), 'output/netD_Y.pkl')
