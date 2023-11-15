import logging
from typing import Dict, AnyStr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.utils as vutils
import numpy as np


class VariationalAE_Flat128d(nn.Module):

    def __init__(self,
                 device: torch.device,
                 input_dim: int = 128 * 128,
                 hidden_dim_1: int = 4096,
                 hidden_dim_2: int = 1024,
                 hidden_dim_3: int = 512,
                 latent_dim: int = 256):
        super(VariationalAE_Flat128d, self).__init__()

        self.device = device

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim_3, latent_dim),
            nn.LeakyReLU(0.2),
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim_3),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim_3, hidden_dim_2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim_1, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparam(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparam(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mean, log_var


def vae_loss_function(x, x_hat, mean, log_var):
    if not any(torch.isfinite(x_hat.flatten())):
        logging.error('X Hat Value')
        logging.error(x_hat)

    if not any(torch.isfinite(x.flatten())):
        logging.error('X Value')
        logging.error(x)

    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def vae_train(dataloader: DataLoader, configuration: Dict, device: torch.device) -> VariationalAE_Flat128d:
    model = VariationalAE_Flat128d(
        input_dim=configuration['input_dim'],
        hidden_dim_1=configuration['h1_dim'],
        hidden_dim_2=configuration['h2_dim'],
        hidden_dim_3=configuration['h3_dim'],
        latent_dim=configuration['latent_dim'],
        device=device
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    model.train()
    logging.info("Starting Training Loop...")
    for epoch in range(configuration['epoch']):
        overall_loss = 0
        batch_idx = 0
        for batch_idx, x in enumerate(dataloader):
            if len(x) != configuration['batch_size']:
                continue

            x = x.unsqueeze(0).view(
                configuration['batch_size'],
                configuration['input_dim']
            ).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = vae_loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            logging.info(
                "[%d/%d] | Avg. Loss: %.4f |" %
                (epoch + 1, configuration['epoch'],
                 overall_loss / (batch_idx * configuration['batch_size']))
            )

    model.eval()
    logging.info("Ending Training...")
    return model


def vae_generate_latent_space_out(
        model: VariationalAE_Flat128d,
        target_dir: AnyStr,
        device: torch.device,
        scale: float = 5.0,
        n: int = 25,
        image_size: int = 128
):
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    count = 0
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model.decode(z_sample)

            image = x_decoded[0].detach().cpu().reshape(image_size, image_size)
            vutils.save_image(image, f'{target_dir}/vae_{count}.png')
            logging.info(f'Saved: {target_dir}/vae_{count}.png')
            count += 1

    return
