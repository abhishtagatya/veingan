import logging
from typing import List, Dict, AnyStr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.utils as vutils


class DCGenerator_128d(nn.Module):
    def __init__(self, ngpu: int = 0, nz: int = 100, ngf: int = 128, nc: int = 3):
        super(DCGenerator_128d, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. ``(ngf*16) x 4 x 4``
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 8 x 8``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 16 x 16``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 32 x 32``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 64 x 64``
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 128 x 128``
        )

    def forward(self, x):
        return self.main(x)


class DCDiscriminator_128d(nn.Module):
    def __init__(self, ngpu: int = 0, ndf: int = 128, nc: int = 3):
        super(DCDiscriminator_128d, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 128 x 128``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. ``(ndf) x 64 x 64``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. ``(ndf*2) x 32 x 32``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. ``(ndf*4) x 16 x 16``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. ``(ndf*8) x 8 x 8``
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # state size. ``(ndf*16) x 4 x 4``
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
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


def dcgan_prepare(configuration: Dict, device: torch.device) -> (DCGenerator_128d, DCDiscriminator_128d):
    netG = DCGenerator_128d(
        ngpu=configuration['ngpu'],
        ngf=configuration['ngf'],
        nz=configuration['nz'],
        nc=configuration['nc']
    ).to(device)
    netD = DCDiscriminator_128d(
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
    fixed_noise = torch.randn(128, configuration['nz'], 1, 1, device=device)

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
                logging.info('[%d/%d][%d/%d] | D: %.4f G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f |'
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


class CG_ConvolutionalBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            is_downsampling: bool = True,
            add_activation: bool = True,
            **kwargs
    ):
        super(CG_ConvolutionalBlock, self).__init__()
        if is_downsampling:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )
        else:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )

    def forward(self, x):
        return self.main(x)


class CG_ResidualBlock(nn.Module):

    def __init__(self, channels: int):
        super(CG_ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            CG_ConvolutionalBlock(channels, channels, add_activation=True, kernel_size=3, padding=1),
            CG_ConvolutionalBlock(channels, channels, add_activation=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.main(x)


class CG_ConvINormPacked(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(CG_ConvINormPacked, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class CGGenerator_128d(nn.Module):

    def __init__(self, img_channels: int, num_features: int = 64, num_residuals: int = 9):
        super(CGGenerator_128d, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
        )

        self.downsampling_layers = nn.ModuleList(
            [
                CG_ConvolutionalBlock(
                    in_channels=num_features,
                    out_channels=num_features * 2,
                    is_downsampling=True,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                CG_ConvolutionalBlock(
                    in_channels=num_features * 2,
                    out_channels=num_features * 4,
                    is_downsampling=True,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.residual_layers = nn.Sequential(
            *[CG_ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        self.upsampling_layers = nn.ModuleList(
            [
                CG_ConvolutionalBlock(
                    in_channels=num_features * 4,
                    out_channels=num_features * 2,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                CG_ConvolutionalBlock(
                    in_channels=num_features * 2,
                    out_channels=num_features * 1,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last_layer = nn.Conv2d(
            in_channels=num_features * 1,
            out_channels=img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.downsampling_layers:
            x = layer(x)
        x = self.residual_layers(x)
        for layer in self.upsampling_layers:
            x = layer(x)
        return torch.tanh(self.last_layer(x))


class CGDiscriminator_128d(nn.Module):

    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(CGDiscriminator_128d, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CG_ConvINormPacked(
                    in_channels=in_channels,
                    out_channels=feature,
                    stride=1 if feature == features[-1] else 2,
                )
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)
        return torch.sigmoid(self.model(x))


def cyclegan_prepare(
        configuration: Dict, device: torch.device
) -> (CGDiscriminator_128d, CGDiscriminator_128d, CGGenerator_128d, CGGenerator_128d):
    d_X = CGDiscriminator_128d(in_channels=configuration['nc']).to(device)
    d_Y = CGDiscriminator_128d(in_channels=configuration['nc']).to(device)
    g_X = CGGenerator_128d(img_channels=configuration['nc'], num_residuals=configuration['nr']).to(device)
    g_Y = CGGenerator_128d(img_channels=configuration['nc'], num_residuals=configuration['nr']).to(device)

    return d_X, d_Y, g_X, g_Y


def cyclegan_save(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        save_as: AnyStr = 'model.pth.tar'
):
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    return torch.save(checkpoint, save_as)


def cyclegan_load(
        checkpoint_file: AnyStr,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        lr: float
):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return model, optimizer


def cyclegan_train(dataloader: DataLoader, target_dir: AnyStr, configuration: Dict, device: torch.device):
    d_X, d_Y, g_X, g_Y = cyclegan_prepare(configuration, device)
    opt_D = Adam(list(d_X.parameters()) + list(d_Y.parameters()), lr=configuration['lr_D'],
                 betas=(configuration['beta1'], 0.999))
    opt_G = Adam(list(g_X.parameters()) + list(g_Y.parameters()), lr=configuration['lr_G'],
                 betas=(configuration['beta1'], 0.999))

    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    scaler_D = torch.cuda.amp.GradScaler()
    scaler_G = torch.cuda.amp.GradScaler()

    logging.info('Start Training Loop...')
    for epoch in range(configuration['epoch']):
        X_reals, X_fakes = 0, 0
        Y_reals, Y_fakes = 0, 0
        for i, (x, y) in enumerate(dataloader, 0):
            x = x.to(device)  # Finger
            y = y.to(device)  # Support

            with torch.cuda.amp.autocast():
                fake_y = g_Y(x)
                d_Y_real = d_Y(y)
                d_Y_fake = d_Y(fake_y.detach())
                Y_reals += d_Y_real.mean().item()
                Y_fakes += d_Y_fake.mean().item()
                d_Y_real_loss = criterion_mse(d_Y_real, torch.ones_like(d_Y_real))
                d_Y_fake_loss = criterion_mse(d_Y_fake, torch.zeros_like(d_Y_fake))
                d_Y_loss = d_Y_real_loss + d_Y_fake_loss

                fake_x = g_X(y)
                d_X_real = d_X(x)
                d_X_fake = d_X(fake_x.detach())
                X_reals += d_X_real.mean().item()
                X_fakes += d_X_fake.mean().item()
                d_X_real_loss = criterion_mse(d_X_real, torch.ones_like(d_X_real))
                d_X_fake_loss = criterion_mse(d_X_fake, torch.zeros_like(d_X_fake))
                d_X_loss = d_X_real_loss + d_X_fake_loss

                d_loss = (d_Y_loss + d_X_loss) / 2

            opt_D.zero_grad()
            scaler_D.scale(d_loss).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            with torch.cuda.amp.autocast():
                d_Y_fake = d_Y(fake_y)
                d_X_fake = d_X(fake_x)
                g_Y_loss = criterion_mse(d_Y_fake, torch.ones_like(d_Y_fake))
                g_X_loss = criterion_mse(d_X_fake, torch.ones_like(d_X_fake))

                cycle_X = g_X(fake_y)
                cycle_Y = g_Y(fake_x)
                cycle_X_loss = criterion_l1(x, cycle_X)
                cycle_Y_loss = criterion_l1(y, cycle_Y)

                identity_X = g_X(x)
                identity_Y = g_Y(y)
                identity_X_loss = criterion_l1(x, identity_X)
                identity_Y_loss = criterion_l1(y, identity_Y)

                g_loss = (
                        g_X_loss + g_Y_loss
                        + (cycle_X_loss * configuration['lambda_cycle'])
                        + (cycle_Y_loss * configuration['lambda_cycle'])
                        + (identity_X_loss * configuration['lambda_identity'])
                        + (identity_Y_loss * configuration['lambda_identity'])
                )

            opt_G.zero_grad()
            scaler_G.scale(g_loss).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            if i % 600 == 0:
                logging.info(
                    '[%d/%d][%d/%d] | G_X: %.4f D_X: %.4f G_Y: %.4f D_Y: %.4f G: %.4f D: %.4f |'
                    % (
                        epoch, configuration['epoch'], i, len(dataloader),
                        g_X_loss, d_X_loss, g_Y_loss, d_Y_loss,
                        g_loss, d_loss
                    )
                )
                vutils.save_image(fake_x * 0.5 + 0.5, f"{target_dir}/finger_{i}.png")
                vutils.save_image(fake_y * 0.5 + 0.5, f"{target_dir}/support_{i}.png")

        if configuration['save_model']:
            cyclegan_save(g_X, opt_G, save_as='./pretrained/cyclegan/gX.pth.tar')
            cyclegan_save(g_Y, opt_G, save_as='./pretrained/cyclegan/gY.pth.tar')
            cyclegan_save(d_X, opt_D, save_as='./pretrained/cyclegan/dX.pth.tar')
            cyclegan_save(d_Y, opt_D, save_as='./pretrained/cyclegan/dY.pth.tar')

    logging.info('End Training...')
    return


def cyclegan_infer(dataloader: DataLoader, target_dir: AnyStr, configuration: Dict, device: torch.device):
    _, _, g_X, g_Y = cyclegan_prepare(configuration, device)
    opt_G = Adam(list(g_X.parameters()) + list(g_Y.parameters()), lr=configuration['lr_G'],
                 betas=(configuration['beta1'], 0.999))

    g_X, opt_GX = cyclegan_load(configuration['gX_ckpt'], g_X, opt_G, device, configuration['lr'])
    g_Y, opt_GY = cyclegan_load(configuration['gY_ckpt'], g_Y, opt_G, device, configuration['lr'])

    logging.info('Starting Infer Mode...')
    for i, (x, y) in enumerate(dataloader, 0):
        x = x.to(device)  # Finger
        y = y.to(device)  # Support

        with torch.no_grad():
            fake_y = g_Y(x)
            fake_x = g_X(y)
            vutils.save_image(fake_x * 0.5 + 0.5, f"{target_dir}/finger_{i}.png")
            vutils.save_image(fake_y * 0.5 + 0.5, f"{target_dir}/support_{i}.png")

    logging.info('Ending Infer...')
    return
