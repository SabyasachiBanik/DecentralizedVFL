# DecentralizedVFL

## Introduction

With the rapid growth of data generated from IoT devices, Artificial Intelligence (AI), particularly Machine Learning (ML), is poised to transform data-driven decision-making. However, traditional centralized machine learning models face significant challenges, particularly in data processing and privacy. Federated Learning (FL) has emerged as a solution, allowing collaborative model training across multiple devices without the need to centralize data. This project explores Vertical Federated Learning (VFL) within a decentralized environment. VFL is particularly suited for scenarios where multiple parties hold different features of the same dataset. This work introduces three distinct prototypes to showcase the capabilities and advantages of decentralized VFL.


## Prototypes Overview

### Prototype 1: Initial VFL Setup Architecture
- **Data Distribution Methodology:** Features are distributed among participants using a round-robin method, ensuring a balanced and diverse distribution.

  <img width="555" alt="image" src="https://github.com/user-attachments/assets/ecf33542-b54c-4930-9015-e123b0e6e411">

- **Global Model Architecture:** A shared global model architecture is agreed upon by all participants. Each participant trains their local model, shares gradients, and aggregates them in a decentralized manner.
- **Key Feature:** Zero-padding is used to handle missing features during training, ensuring that all participants contribute to the overall model.

### Prototype 2: Enhanced VFL Setup with Hidden Layer Output Sharing
- **Hidden Layer Output Sharing:** In addition to gradients, participants share the outputs of their hidden layers. This additional sharing improves the learning process by providing more comprehensive intermediate representations.


### Prototype 3: Advanced VFL with Hidden Layer Output and Backpropagation Loss Sharing
- **Backpropagation Loss Sharing:** Building upon Prototype 2, this version includes the sharing of backpropagation losses. This further aligns participants in their understanding of the loss landscape during training.


## Evaluation

The performance of the prototypes was evaluated on several datasets including MNIST, Fashion MNIST, Titanic, Adult, bank Marketing and Give Me Some Credit. Our decentralized VFL approach outperformed or proved competitive in comparison to the performance of state-of-the-art methods across these datasets.

## How to Reproduce the Experiments

### Requirements
To reproduce the experiments, the insatllation of the packages declared in the file requirements.txt needs to be ensured.

### Running the Code
1. **Clone the Repository:**
   
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
