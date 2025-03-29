
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
import math

from VAE import VAE, vae_loss
from Classifier import Classifier
from utils import imshow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Working on device: {device}")
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 16
TYPE = "Classifier on private data"
PRIVATE = True
EPSILON = 50
DELTA = 1e-5
MAX_GRAD_NORM = 1.2
NOISE_MULTIPLIER = 0.1
LOG = False

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
                   "max_grad_norm": MAX_GRAD_NORM,
                   "noise_multiplier": NOISE_MULTIPLIER,
                   "vae_mean": "no mean"

               }
               )


def train_classifier(train_loader, test_loader, save_path, private=False):
    classifier = Classifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if private:
        privacy_engine = PrivacyEngine()
        classifier, optimizer, train_loader = privacy_engine.make_private(
            module=classifier,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=NOISE_MULTIPLIER,
            max_grad_norm=MAX_GRAD_NORM
        )

    print(f"Training classifier private={private}.....")

    # classifier.train()
    for epoch in range(EPOCHS):
        classifier.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss += loss.item()

            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            if math.isnan(epoch_loss):
                print("Nan Loss")
                print(loss)
                print(loss.item())
                # imshow(torchvision.utils.make_grid(images.to('cpu')), title="original")
                print(images)
                print(labels)
                return

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




    # torch.save(classifier, f"./Trained/mnist_classifier_private_{private}_2.pth")
    torch.save(classifier, f"./Trained/{save_path}.pth")

    print(f"Finished training classifier, private={private}")



def train_vae(train_loader, test_loader, private=True):
    # Initialize model, optimizer, and privacy engine
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    if private:
        privacy_engine = PrivacyEngine()
        vae, optimizer, train_loader = privacy_engine.make_private(
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

        if LOG: wandb.log({"test_loss": test_loss / len(test_loader),
                           # "epoch": epoch + 1,
                           # "loss": epoch_loss / len(train_loader)
                           })
        print(f"Test Loss: {test_loss / len(test_loader):.4f}")

    torch.save(vae.state_dict(), f"./Trained/mnist_vae_private_{private}_noise_{NOISE_MULTIPLIER}_no_means_state_dict_2.pth")

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


def generate_synthetic_data(vae: VAE, num_samples=10000):
    with torch.no_grad():
        labels = torch.randint(0, 10, (num_samples,)).to(device)
        z = torch.randn(num_samples, vae.latent_dim).to(device)
        z = z + vae.class_means[labels]
        synthetic_images = vae.decoder(z).view(-1, 1, 28, 28).cpu()
    return synthetic_images, labels

def generate_synthetic_data_with_mean_calculation(vae: VAE, train_loader, num_samples_per_class=1000):
    class_means = torch.zeros(10, vae.latent_dim).to(device)
    class_stds = torch.zeros(10, vae.latent_dim).to(device)
    class_counts = torch.zeros(10).to(device)

    vae.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            _, mu, logvar, _ = vae(images, labels)
            std = torch.exp(0.5 * logvar)
            for i in range(10):
                class_mask = (labels == i)
                class_means[i] += mu[class_mask].sum(dim=0)
                class_stds[i] += std[class_mask].sum(dim=0)
                class_counts[i] += class_mask.sum()

    class_means /= class_counts.unsqueeze(1)
    class_stds /= class_counts.unsqueeze(1)

    # Generate random vectors for each class and decode to generate images
    generated_images = []

    for i in range(10):
        z = class_means[i] + class_stds[i] * torch.randn(num_samples_per_class, vae.latent_dim).to(device)
        generated_images.append(vae.decoder(z).view(-1, 1, 28, 28).cpu())

    generated_images = torch.cat(generated_images, dim=0)
    labels = torch.cat([torch.full((num_samples_per_class,), i, dtype=torch.long) for i in range(10)], dim=0)

    return generated_images, labels

def plot_synthetic_data(
        private_dataset,
        private_labels,
        # private_vae_path,
        images_per_class = 2):
    # vae = VAE().to(device)
    # print("loading vae....")
    # state_dict = torch.load(private_vae_path, weights_only=False)
    # new_state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
    # vae.load_state_dict(new_state_dict, strict=True)
    # vae = torch.load(private_vae_path, weights_only=False)
    # vae.eval()
    # private_dataset, private_labels = generate_synthetic_data(vae, num_samples=10000)
    private_dataset = private_dataset.to("cpu")
    private_labels = private_labels.to("cpu")
    num_classes = 10
    selected_images = []
    for class_idx in range(num_classes):
        class_images = private_dataset[private_labels == class_idx][:images_per_class]
        selected_images.append(class_images)

    # Concatenate the selected images
    selected_images = torch.cat(selected_images, dim=0)

    # Display the images
    imshow(torchvision.utils.make_grid(selected_images.to("cpu"), nrow=images_per_class),
           title=f"Synthetic Images generated by VAE with noise {NOISE_MULTIPLIER} ({images_per_class} per class)")

def run():
    train_loader, test_loader = init()

    # train the classifier non-private
    train_classifier(train_loader, test_loader,
                     f"mnist_classifier_non_private", private=False)
    #

    private_classifier_path = "Trained/PrivateClassifier.pth"
    private_vae_path = "Trained/mnist_vae_private_True_state_dict.pth"
    classifier_path = "Trained/Classifier.pth"

    # Train the classifier with private guarantee
    train_classifier(train_loader,
                     test_loader,
                     f"mnist_classifier_PRIVATE_noise_{NOISE_MULTIPLIER}",
                     private=True)


    # Private VAE and regular classifier
    # Train the VAE with private guarantee and the classifier regularlly
    train_vae(train_loader, test_loader, private=True)
    # return



    noise_levels = [0.03, 0.06, 0.1]
    for noise in noise_levels:
    #     noise = NOISE_MULTIPLIER
        print("working with noise: ", noise)
        save_path = f"mnist_classifier_private_dataset_noise_{noise}_vae_no_means"
        # private_vae_path = f"Trained/mnist_vae_private_True_noise_{noise}_state_dict_2.pth"
        private_vae_path = f"Trained/mnist_vae_private_True_noise_{noise}_no_means_state_dict_2.pth"
        vae = VAE().to(device)
        print("loading vae....")
        state_dict = torch.load(private_vae_path, weights_only=False)
        new_state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
        vae.load_state_dict(new_state_dict, strict=True)
        # vae = torch.load(private_vae_path, weights_only=False)
        vae.eval()
        print("generating synthetic data....")
        # private_dataset, private_labels = generate_synthetic_data(vae, num_samples=10000)
        private_dataset, private_labels = generate_synthetic_data_with_mean_calculation(vae, train_loader)
        plot_synthetic_data(private_dataset, private_labels, images_per_class=5)
        # return
        # plot_synthetic_data(private_dataset, private_labels, images_per_class=2)
        # imshow(torchvision.utils.make_grid(private_dataset[:20].to('cpu')),
        #        title="generated private images")
        torch.save(private_dataset, f"synthetic_mnist_noise_{noise}.pth")
        torch.save(private_labels, f"synthetic_mnist_labels_noise_{noise}.pth")


        private_loader = DataLoader(list(zip(private_dataset, private_labels)), batch_size=BATCH_SIZE, shuffle=True)
        train_classifier(private_loader,test_loader,save_path, private=False)


    # load the synthetic and plot 20 images 2 from each class
    # plot_synthetic_data(private_vae_path, images_per_class=2)




if __name__ == '__main__':
    run()