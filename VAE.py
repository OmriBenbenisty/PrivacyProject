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
        # self.class_means = nn.Parameter(torch.randn(num_classes, latent_dim))
        self.class_means = torch.randn(num_classes, latent_dim).to(device)
        # self.class_means = torch.randn(num_classes, latent_dim).cpu().numpy()
        # self.class_logvars = nn.Parameter(torch.randn(num_classes, latent_dim))
        self.class_logvars = torch.randn(num_classes, latent_dim).to(device)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        # if torch._C._functorch.is_grad_tracking_in_vmap_mode():
        #     eps = torch.func.vmap(lambda x: torch.randn_like(x), randomness="different")(std)
        # else:
        #     eps = torch.randn_like(std)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels=None):
        x_encoded = self.encoder(x)
        mu, logvar = x_encoded[:, :self.latent_dim], x_encoded[:, self.latent_dim:]
        labels = None
        if labels is not None:
            class_mu = torch.index_select(self.class_means, 0, labels)
            # class_logvar = self.class_logvars[labels]
            class_logvar = torch.index_select(self.class_logvars, 0, labels)
            mu = mu + class_mu
            logvar = logvar + class_logvar

        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        x_reconstructed = x_reconstructed.reshape(-1, 28, 28)
        # if x_reconstructed.ndim == 3:
        #     x_reconstructed = x_reconstructed.unsqueeze(1)

        return x_reconstructed, mu, logvar, z

# Loss function with Gaussian regularization
def vae_loss(recon_x, x, mu, logvar):
    if recon_x.shape != x.shape:
        print("recon_x.shape: ", recon_x.shape)
        print("x.shape: ", x.shape)
        assert False
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


