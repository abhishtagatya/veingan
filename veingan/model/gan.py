import logging
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam


class DCGenerator_64d(nn.Module):
    def __init__(self, ngpu: int = 0, nz: int = 100, ngf: int = 64, nc: int = 3):
        super(DCGenerator_64d, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, x):
        return self.main(x)


class DCDiscriminator_64d(nn.Module):
    def __init__(self, ngpu: int = 0, ndf: int = 64, nc: int = 3):
        super(DCDiscriminator_64d, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


def dcgan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def dcgan_prepare(configuration: Dict, device: torch.device) -> (DCGenerator_64d, DCDiscriminator_64d):
    netG = DCGenerator_64d(
        ngpu=configuration['ngpu'],
        ngf=configuration['ngf'],
        nz=configuration['nz'],
        nc=configuration['nc']
    ).to(device)
    netD = DCDiscriminator_64d(
        ngpu=configuration['ngpu'],
        ndf=configuration['ndf'],
        nc=configuration['nc']
    ).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (configuration['ngpu'] > 1):
        netG = nn.DataParallel(netG, list(range(configuration['ngpu'])))
        netD = nn.DataParallel(netD, list(range(configuration['ngpu'])))

    netG.apply(dcgan_weights_init)
    netD.apply(dcgan_weights_init)

    return netG, netD


def dcgan_train(dataloader: DataLoader, configuration: Dict, device: torch.device) -> List:
    netG, netD = dcgan_prepare(configuration, device)
    fixed_noise = torch.randn(64, configuration['nz'], 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    criterion = nn.BCELoss()
    optimizerD = Adam(netD.parameters(), lr=configuration['lr_D'], betas=(configuration['beta1'], 0.999))
    optimizerG = Adam(netG.parameters(), lr=configuration['lr_G'], betas=(configuration['beta1'], 0.999))

    # Trackers
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    logging.info("Starting Training Loop...")
    # For each epoch
    for epoch in range(configuration['epoch']):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].unsqueeze(0).to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, configuration['nz'], 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 32 == 0:
                logging.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                             % (epoch, configuration['epoch'], i, len(dataloader),
                                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == configuration['epoch'] - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(fake)

            iters += 1

    logging.info("Ending Training...")
    return img_list
