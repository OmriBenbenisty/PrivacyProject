import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
import numpy as np

from VAE import VAE, vae_loss
from Classifier import Classifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_classifier():
    classifier = Classifier()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    classifier.train()
    for epoch in range(5):
        epoch_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(torch.device('cpu')), labels.to(torch.device('cpu'))
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch [{epoch+1}/5], Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")
    torch.save(classifier, "mnist_classifier.pth")



def train_vae():
    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize model, optimizer, and privacy engine
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    privacy_engine = PrivacyEngine()
    vae, optimizer, data_loader = privacy_engine.make_private(
        module=vae,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0
    )

    # Training loop
    num_epochs = 10
    vae.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, labels in data_loader:
            images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar, _ = vae(images, labels)
            loss = vae_loss(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataset):.4f}")

    # Generate synthetic dataset
    def generate_synthetic_data(num_samples=10000):
        with torch.no_grad():
            labels = torch.randint(0, 10, (num_samples,)).to(device)
            z = torch.randn(num_samples, vae.latent_dim).to(device)
            z = z + vae.class_means[labels]
            synthetic_images = vae.decoder(z).view(-1, 1, 28, 28).cpu()
        return synthetic_images

    synthetic_data = generate_synthetic_data()
    torch.save(synthetic_data, "synthetic_mnist.pth")

if __name__ == '__main__':
    train_classifier()