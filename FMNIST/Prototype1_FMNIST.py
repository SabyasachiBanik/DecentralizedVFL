import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import random

# Reproducibility with seeding
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Fashion-MNIST data
def load_fashion_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

# Robin-Round Vertical Split
def vertical_partition_rotating(dataset, num_participants):
    partitions = [[] for _ in range(num_participants)]
    num_rows = dataset[0][0].shape[1]

    for image, label in dataset:
        full_images = [torch.zeros_like(image) for _ in range(num_participants)] # Each image is initialized as a zero tensor with the same shape as the original image, effectively zero-padding all features initially. Only the features corresponding to each participant are filled with actual data, while the rest remain as zeros.
        for i in range(num_rows):
            participant = i % num_participants
            row_data = image[:, i, :].unsqueeze(1)
            full_images[participant][:, i, :] = row_data.squeeze(1)

        for participant in range(num_participants):
            partitions[participant].append((full_images[participant], label))

    for i in range(num_participants):
        data_tensors, labels = zip(*partitions[i])
        partitions[i] = TensorDataset(torch.stack(data_tensors), torch.tensor(labels))
    return partitions



class GlobalModel(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, output_size):
        super(GlobalModel, self).__init__()
        self.segments = nn.ModuleList()
        for input_size, hidden_size in zip(input_sizes, hidden_sizes):
            layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size)]
            self.segments.append(nn.Sequential(*layers))

    def forward(self, x, active_segments):
        segment_outputs = []
        start_index = 0
        for i, segment in enumerate(self.segments):
            end_index = start_index + input_sizes[i]
            if i in active_segments:
                segment_input = x[:, start_index:end_index]  # We extract the segment of the input corresponding to the active participant's features.
                segment_output = segment(segment_input)
                segment_outputs.append(segment_output)
            else:
                segment_outputs.append(torch.zeros(x.size(0), output_size, device=x.device)) # Segments not corresponding to the active participant are replaced with zero tensors
            start_index = end_index
        combined_output = torch.mean(torch.stack(segment_outputs), dim=0) 
        return combined_output

# Training function
def train(model, device, train_loader, optimizer, epoch, input_sizes, participant_id):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1) # We flatten the image into a single vector, preserving the zero values for non-responsible features
        optimizer.zero_grad()
        output = model(data, active_segments=[participant_id])
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

# Function to safely initialize or get gradients
def safe_gradient(parameter):
    if parameter.grad is not None:
        return parameter.grad.data
    else:
        return torch.zeros_like(parameter.data)

# Selective gradient exchange function
def selective_exchange_gradients(models, input_sizes, hidden_sizes):
    num_models = len(models)
    param_indices = [0]
    cumulative_index = 0
    for i in range(len(hidden_sizes)):
        cumulative_index += (input_sizes[i] * hidden_sizes[i]) + hidden_sizes[i]
        param_indices.append(cumulative_index)

    for seg in range(len(hidden_sizes)):
        start = param_indices[seg]
        end = param_indices[seg + 1]
        for param_idx in range(start, end):
            grads = []
            for model in models:
                model_params = list(model.parameters())
                if param_idx < len(model_params) and model_params[param_idx].grad is not None:
                    grads.append(model_params[param_idx].grad)
            if grads:
                avg_grad = torch.stack(grads).mean(dim=0)
                for model in models:
                    model_params = list(model.parameters())
                    if param_idx < len(model_params):
                        model_params[param_idx].grad = avg_grad.clone()
    final_start = param_indices[-1]
    if final_start < len(list(models[0].parameters())):
        for param_idx in range(final_start, len(list(models[0].parameters()))):
            grads = []
            for model in models:
                model_params = list(model.parameters())
                if param_idx < len(model_params) and model_params[param_idx].grad is not None:
                    grads.append(model_params[param_idx].grad)
            if grads:
                avg_grad = torch.stack(grads).mean(dim=0)
                for model in models:
                    if param_idx < len(list(model.parameters())):
                        list(model.parameters())[param_idx].grad = avg_grad.clone()

# Evaluation function
def evaluate(models, device, test_loader):
    with torch.no_grad():
        test_loss = 0
        correct = 0
        all_preds = []
        all_targets = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            outputs = torch.zeros(data.size(0), 10, device=device)
            for model in models:
                output = model(data, active_segments=list(range(len(model.segments))))
                outputs += output
            outputs /= len(models)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(outputs, target).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.view(-1).cpu().numpy())

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy

# Main block for execution
train_dataset, test_dataset = load_fashion_mnist_data()

num_seeds = 5
participant_accuracies = {i: [] for i in range(2, 11)}

for seed in range(num_seeds):
    set_seed(seed)

    for num_participants in range(2, 11):  # 2 to 10 participants
        partitioned_datasets = vertical_partition_rotating(train_dataset, num_participants)
        input_sizes = [784 // num_participants] * (num_participants - 1) + [784 - (num_participants - 1) * (784 // num_participants)]
        hidden_sizes = [10] * num_participants  # For Prototype 1, we take num_participants = numner of hidden layers
        output_size = 10

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = [GlobalModel(input_sizes, hidden_sizes, output_size).to(device) for _ in range(num_participants)]
        optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

        federated_rounds = 5
        epochs_per_round = 5

        for federated_round in range(federated_rounds):
            for participant_id in range(num_participants):
                train_loader = DataLoader(partitioned_datasets[participant_id], batch_size=64, shuffle=True)
                for epoch in range(1, epochs_per_round + 1):
                    train(models[participant_id], device, train_loader, optimizers[participant_id], epoch, input_sizes, participant_id)

            selective_exchange_gradients(models, input_sizes, hidden_sizes)

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        accuracy = evaluate(models, device, test_loader)
        participant_accuracies[num_participants].append(accuracy)
        print(f"Seed: {seed}, Number of Participants: {num_participants}, Accuracy: {accuracy:.2f}%")

# Calculate the average accuracy for each number of participants
avg_accuracies = {num_participants: np.mean(accs) for num_participants, accs in participant_accuracies.items()}

"""
# We plot all accuracies in a separate file - Accuracy Plots
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), [avg_accuracies[i] for i in range(2, 11)], marker='o', linestyle='-', color='b')
plt.xlabel('Number of Participants')
plt.ylabel('Accuracy (%)')
plt.title('Average Accuracy vs. Number of Participants')
plt.grid(True)
plt.show()
"""