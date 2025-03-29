# Differential Privacy in Machine Learning with VAEs

This project explores two approaches to integrating differential privacy (DP) into machine learning workflows: training a classifier with differential privacy guarantees and generating a differentially private synthetic dataset to train a classifier. The project utilizes Variational Autoencoders (VAEs) and the Opacus library to ensure privacy during the training process.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In this study, we explore two approaches to integrating differential privacy into machine learning workflows:
1. Training a classifier with differential privacy guarantees.
2. Generating a differentially private synthetic dataset to train a classifier.

We use Variational Autoencoders (VAEs) and the Opacus library to ensure privacy during the training process. The first approach involves directly training a classifier with differential privacy, ensuring that the model parameters do not leak sensitive information. The second approach involves training a VAE with differential privacy to generate a synthetic dataset that mimics the original data distribution, which is then used to train a classifier without privacy constraints.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt