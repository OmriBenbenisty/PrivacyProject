import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
            nn.Linear(400, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()
        )
        self.latent_dim = latent_dim
        self.class_means = nn.Parameter(torch.randn(num_classes, latent_dim))
        self.class_logvars = nn.Parameter(torch.randn(num_classes, latent_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        x_encoded = self.encoder(x)
        mu, logvar = x_encoded[:, :self.latent_dim], x_encoded[:, self.latent_dim:]
        class_mu = self.class_means[labels]
        class_logvar = self.class_logvars[labels]
        mu = mu + class_mu
        logvar = logvar + class_logvar
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar, z

# Loss function with Gaussian regularization
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x.view(-1, 28 * 28), reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


