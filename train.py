
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from matplotlib.pyplot import title
from torch.utils.data import DataLoader, random_split
from opacus import PrivacyEngine
import numpy as np
import wandb

from VAE import VAE, vae_loss
from Classifier import Classifier
from utils import imshow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Working on device: {device}")
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 4
TYPE = "VAE"
PRIVATE = True
EPSILON = 50
DELTA = 1e-5
MAX_GRAD_NORM = 1.2
NOISE_MULTIPLIER = 0.05
LOG = True

if LOG:
    wandb.init(project="PrivacyVAE",
               config={
                   "learning_rate": LEARNING_RATE,
                   "dataset": "MNIST",
                   "epochs": EPOCHS,
                   "batch_size": BATCH_SIZE,
                   "name": f"{TYPE}_{PRIVATE}",
                   "type": TYPE,
                   "private": PRIVATE,
                   "epsilon": EPSILON,
                   "delta": DELTA,
                   "max_grad_norm": MAX_GRAD_NORM

               }
               )


def train_classifier(train_loader, test_loader, private=False):
    classifier = Classifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if private:
        privacy_engine = PrivacyEngine()
        classifier, optimizer, train_loader = privacy_engine.make_private(
            module=classifier,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=1.1,
            max_grad_norm=1.0
        )

    print(f"Training classifier private={private}.....")

    classifier.train()
    for epoch in range(EPOCHS):
        classifier.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(torch.device(device)), labels.to(torch.device(device))
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        if LOG: wandb.log({"epoch": epoch + 1, "loss": epoch_loss / len(train_loader), "accuracy": accuracy})
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")



        # Evaluate on test_loader
        test_loss = 0
        test_correct = 0
        test_total = 0
        classifier.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = classifier(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_correct += (outputs.argmax(dim=1) == labels).sum().item()
                test_total += labels.size(0)
        test_accuracy = test_correct / test_total
        if LOG: wandb.log({"test_loss": test_loss / len(test_loader), "test_accuracy": test_accuracy})
        print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.4f}")




    torch.save(classifier, f"./Trained/mnist_classifier_private_{private}.pth")

    print(f"Finished training classifier, private={private}")



def train_vae(train_loader, test_loader, private=True):
    # Initialize model, optimizer, and privacy engine
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    if private:
        privacy_engine = PrivacyEngine()
        vae, optimizer, data_loader = privacy_engine.make_private(
            module=vae,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=NOISE_MULTIPLIER,
            max_grad_norm=MAX_GRAD_NORM,
            # target_delta=DELTA,
            # target_epsilon=EPSILON,
            # epochs=EPOCHS,
            # functorch_grad_sample_mode="hooks"
        )

    # Training loop
    print("Training VAE.....")
    vae.train()
    for epoch in range(EPOCHS):
        vae.train()
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.view(-1, 1, 28, 28).to(device), labels.to(device)
            optimizer.zero_grad()
            # recon_images, mu, logvar, _ = torch.vmap(vae, randomness='different')(images, labels)
            recon_images, mu, logvar, _ = vae(images, labels)
            loss = vae_loss(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 3000 == 2999:
                imshow(torchvision.utils.make_grid(images.to('cpu')), title="original")
                print("inputs")
                imshow(torchvision.utils.make_grid(recon_images.to('cpu')),title='reosntructed')
                print("outputs")

        if LOG: wandb.log({"epoch": epoch + 1, "loss": epoch_loss / len(train_loader)})
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}")

        # Evaluate on test_loader
        test_loss = 0
        vae.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.view(-1, 1, 28, 28).to(device), labels.to(device)
                # recon_images, mu, logvar, _ = torch.vmap(vae, randomness='different')(images, labels)
                recon_images, mu, logvar, _ = vae(images, labels)
                loss = vae_loss(recon_images, images, mu, logvar)
                test_loss += loss.item()
        if LOG: wandb.log({"test_loss": test_loss / len(test_loader)})
        print(f"Test Loss: {test_loss / len(test_loader):.4f}")

    torch.save(vae, f"./Trained/mnist_vae_private_{private}_2.pth")

    print("Finished training VAE")
    # Generate synthetic dataset



def init():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def generate_synthetic_data(vae, num_samples=10000):
    with torch.no_grad():
        labels = torch.randint(0, 10, (num_samples,)).to(device)
        z = torch.randn(num_samples, vae.latent_dim).to(device)
        z = z + vae.class_means[labels]
        synthetic_images = vae.decoder(z).view(-1, 1, 28, 28).cpu()
    return synthetic_images


def run():
    train_loader, test_loader = init()

    # private_classifier_path = "Trained/PrivateClassifier.pth"
    # vae_path = "Trained/VAE.pth"
    # classifier_path = "Trained/Classifier.pth"

    # Train the classifier with private guarantee
    # train_classifier(train_loader, test_loader, private=True)


    # Train the VAE with private guarantee and the classifier regularlly
    train_vae(train_loader, test_loader, private=PRIVATE)




    # vae = torch.load(vae_path)
    # private_dataset = generate_synthetic_data(vae, num_samples=10000)
    # torch.save(private_dataset, "synthetic_mnist.pth")
    #
    # train_classifier(dataloader, private=False)

if __name__ == '__main__':
    run()