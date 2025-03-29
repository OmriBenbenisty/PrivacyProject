# Differential Privacy in Machine Learning with VAEs

This project explores two approaches to integrating differential privacy (DP) into machine learning workflows: training a classifier with differential privacy guarantees and generating a differentially private synthetic dataset to train a classifier. The project utilizes Variational Autoencoders (VAEs) and the Opacus library to ensure privacy during the training process.

## Table of Contents

- Introduction
- Installation
- Usage
- Project Structure
- Results
- License

## Introduction

In this study, we explore two approaches to integrating differential privacy into machine learning workflows:
1. Training a classifier with differential privacy guarantees.
2. Generating a differentially private synthetic dataset to train a classifier.

We use Variational Autoencoders (VAEs) and the Opacus library to ensure privacy during the training process. The first approach involves directly training a classifier with differential privacy, ensuring that the model parameters do not leak sensitive information. The second approach involves training a VAE with differential privacy to generate a synthetic dataset that mimics the original data distribution, which is then used to train a classifier without privacy constraints.

## Installation

To install the required dependencies, run:

pip install -r requirements.txt

## Usage

To run the project, you can run the train.py file of follow these steps:

1. Initialize the datasets and data loaders:

    train_loader, test_loader = init()

2. Train the VAE with differential privacy:

    train_vae(train_loader, test_loader, private=True)

3. Generate synthetic data using the trained VAE:

    private_dataset, private_labels = generate_synthetic_data_with_mean_calculation(vae, train_loader)

4. Train the classifier on the synthetic dataset:

    train_classifier(private_loader, test_loader, save_path, private=False)

5. Evaluate the results and visualize the synthetic data:

    plot_synthetic_data(private_dataset, private_labels, images_per_class=5)

## Project Structure

.
├── data/                   # Directory for storing datasets
├── Plots/                 # Directory for storing images
├── Trained/                # Directory for storing trained models
├── VAE.py                  # VAE model definition and loss function
├── train.py                # Training and evaluation scripts
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation

## Results

The results of the experiments include the performance of the classifiers and the quality of the synthetic data generated by the VAE. The following metrics are used to evaluate the models:

- Accuracy
- Loss
- Privacy Budget
- Synthetic Data Quality

Example images from the synthetic dataset and loss graphs are included in the Plots/ directory.


## License

This project is licensed under the MIT License. See the LICENSE file for details.