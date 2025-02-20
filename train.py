from idlelib.editor import darwin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
import numpy as np
import wandb

from VAE import VAE, vae_loss
from Classifier import Classifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 64
wandb.init(project="mnist-classification",
           config={
               "learning_rate": LEARNING_RATE,
               "dataset": "MNIST",
               "epochs": EPOCHS,
           }
           )


def train_classifier(data_loader, private=False):
    classifier = Classifier()
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if private:
        privacy_engine = PrivacyEngine()
        classifier, optimizer, data_loader = privacy_engine.make_private(
            module=classifier,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.1,
            max_grad_norm=1.0
        )

    classifier.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        correct = 0
        total = 0
        for images, labels in data_loader:
            images, labels = images.to(torch.device('cpu')), labels.to(torch.device('cpu'))
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / total
        wandb.log({"epoch": epoch + 1, "loss": epoch_loss / len(data_loader), "accuracy": accuracy})
        print(f"Epoch [{epoch+1}/5], Loss: {epoch_loss/len(data_loader):.4f}, Accuracy: {accuracy:.4f}")
    torch.save(classifier, "./Trained/mnist_classifier.pth")



def train_vae(data_loader, private=False):
    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, optimizer, and privacy engine
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    if private:
        privacy_engine = PrivacyEngine()
        vae, optimizer, data_loader = privacy_engine.make_private(
            module=vae,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.1,
            max_grad_norm=1.0
        )

    # Training loop
    vae.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for images, labels in data_loader:
            images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar, _ = vae(images, labels)
            loss = vae_loss(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss / len(dataset):.4f}")

    # Generate synthetic dataset



def init():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return dataset, data_loader


def generate_synthetic_data(vae, num_samples=10000):
    with torch.no_grad():
        labels = torch.randint(0, 10, (num_samples,)).to(device)
        z = torch.randn(num_samples, vae.latent_dim).to(device)
        z = z + vae.class_means[labels]
        synthetic_images = vae.decoder(z).view(-1, 1, 28, 28).cpu()
    return synthetic_images


def run():
    dataset, dataloader = init()

    # private_classifier_path = "Trained/PrivateClassifier.pth"
    # vae_path = "Trained/VAE.pth"
    # classifier_path = "Trained/Classifier.pth"

    # Train the classifier with private guarantee
    train_classifier(dataloader, private=True)


    # Train the VAE with private guarantee and the classifier regularlly
    # train_vae(dataloader, private=True)




    # vae = torch.load(vae_path)
    # private_dataset = generate_synthetic_data(vae, num_samples=10000)
    # torch.save(private_dataset, "synthetic_mnist.pth")
    #
    # train_classifier(dataloader, private=False)

if __name__ == '__main__':
    run()