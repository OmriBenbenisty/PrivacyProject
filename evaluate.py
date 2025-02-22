# import torch
#
# print(torch.cuda.is_available())
# print(torch.version.cuda)



import torch
import numpy as np
import torchvision.utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from VAE import VAE
from utils import imshow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def generate_samples():
    # Load the trained VAE model
    vae = torch.load('Trained/mnist_vae_private_False.pth', weights_only=False)
    vae.eval()


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Calculate the mean and standard deviation of class 3
    class_index = 5
    class_mus = []
    class_stds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            mask = labels == class_index
            if mask.sum() > 0:
                class_images = images[mask]
                x_encoded = vae.encoder(class_images)
                mu, logvar = x_encoded[:, :vae.latent_dim], x_encoded[:, vae.latent_dim:]
                std = torch.exp(0.5 * logvar)
                class_mus.append(mu.cpu().numpy())
                class_stds.append(std.cpu().numpy())

    class_mu = np.concatenate(class_mus, axis=0)
    class_std = np.concatenate(class_stds, axis=0)

    mean = np.mean(class_mu, axis=0)
    std = np.mean(class_std, axis=0)

    # Sample from the latent space using the calculated mean and standard deviation
    num_samples = 10
    z = torch.tensor(np.random.normal(mean, std, size=(num_samples, vae.latent_dim)), dtype=torch.float32).to(device)

    # Use the encoder to generate new samples
    with torch.no_grad():
        generated_images = vae.decoder(z).view(-1, 1, 28, 28).cpu()

    # Save or visualize the generated images
    imshow(torchvision.utils.make_grid(generated_images))



if __name__ == '__main__':
    generate_samples()