# DecentralizedVFL

## Introduction

With the rapid growth of data generated from IoT devices, Artificial Intelligence (AI), particularly Machine Learning (ML), is poised to transform data-driven decision-making. However, traditional centralized machine learning models face significant challenges, particularly in data processing and privacy. Federated Learning (FL) has emerged as a solution, allowing collaborative model training across multiple devices without the need to centralize data. This project explores Vertical Federated Learning (VFL) within a decentralized environment. VFL is particularly suited for scenarios where multiple parties hold different features of the same dataset. This work introduces three distinct prototypes to showcase the capabilities and advantages of decentralized VFL.


## Prototypes Overview

### Prototype 1: Initial VFL Setup Architecture
- **Data Distribution Methodology:** Features are distributed among participants using a round-robin method, ensuring a balanced and diverse distribution.

- **Global Model Architecture:** A shared global model architecture is agreed upon by all participants. Each participant trains their local model, shares gradients, and aggregates them in a decentralized manner.
- **Key Feature:** Zero-padding is used to handle missing features during training, ensuring that all participants contribute to the overall model.


 
### Prototype 2: Enhanced VFL Setup with Hidden Layer Output Sharing
- **Hidden Layer Output Sharing:** In addition to gradients, participants share the outputs of their hidden layers. This additional sharing improves the learning process by providing more comprehensive intermediate representations.

<img width="461" alt="Fig 7i" src="https://github.com/user-attachments/assets/aaffa722-849e-43b7-b607-b5310cfa2fd1">

### Prototype 3: Advanced VFL with Hidden Layer Output and Backpropagation Loss Sharing
- **Backpropagation Loss Sharing:** Building upon Prototype 2, this version includes the sharing of backpropagation losses. This further aligns participants in their understanding of the loss landscape during training.

<img width="441" alt="Fig 7ii" src="https://github.com/user-attachments/assets/5080cff7-1fc5-4e2e-bd20-43a488499a85">


## Evaluation

The performance of the prototypes was evaluated on several datasets including MNIST, Fashion MNIST, Titanic, Adult, bank Marketing and Give Me Some Credit. Our decentralized VFL approach outperformed or proved competitive in comparison to the performance of state-of-the-art methods across these datasets.

## How to Reproduce the Experiments

### Requirements
To reproduce the experiments, the insatllation of the packages declared in the file [requirements.txt](https://github.com/SabyasachiBanik/DecentralizedVFL/blob/main/requirements.txt) needs to be ensured.

### Running the Code
1. **Clone the Repository:**
   
   git clone https://github.com/SabyasachiBanik/DecentralizedVFL.git
   
   cd your_decentralized_vfl

3. **Create a Python Virtual Environment:**

   python3 -m venv venv
   source venv/bin/activate


4. **Install the Required Packages:**

    pip install -r requirements.txt

5. **Run the Experiments:**
    Each dataset folder contains the necessary code to run the experiments. Navigate to the appropriate folder and execute 
    the scripts as needed. Please ensure that the required datasets are available in the correct format as indicated in the 
    code. 

### Conclusion

This project demonstrates the potential of decentralized Vertical Federated Learning (VFL) in handling distributed data across multiple participants. This approach excels in collaborative learning in distributed environments by eliminating the need for a central server and introducing ideal federated mechanisms for sharing intermediate model updates. The results achieved across various datasets highlight the versatility and effectiveness of the VFL prototypes, laying a strong foundation for future research and development in this area.
