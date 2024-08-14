import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import random

# Seeding for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Load MNIST data
def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

# Round-Robin Vertical Splitting
def vertical_partition_rotating(dataset, num_participants):
    partitions = [[] for _ in range(num_participants)]
    num_rows = dataset[0][0].shape[1]
    for image, label in dataset:
        full_images = [torch.zeros_like(image) for _ in range(num_participants)]
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

# Global Model
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
                segment_input = x[:, start_index:end_index]
                segment_output = segment(segment_input)
                segment_outputs.append(segment_output)
            else:
                segment_outputs.append(torch.zeros(x.size(0), output_size, device=x.device))
            start_index = end_index
        combined_output = torch.mean(torch.stack(segment_outputs), dim=0)
        return combined_output

# Training function without gradient exchange
def train_without_federation(model, device, train_loader, optimizer, epoch, input_sizes, participant_id):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        output = model(data, active_segments=[participant_id])
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate(models, device, test_loader):
    total_loss = 0
    total_correct = 0
    total_data = 0

    with torch.no_grad():
        for model in models:
            model.eval()
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = model(data, active_segments=list(range(len(model.segments))))
                loss = nn.CrossEntropyLoss(reduction='sum')(output, target)  # Summing batch loss
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)  # index of the max log-probability is the prediction
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_data += target.size(0)

    average_loss = total_loss / total_data
    accuracy = 100. * total_correct / total_data
    print(f'\nTest set: Average loss: {average_loss:.4f}, Accuracy: {total_correct}/{total_data} ({accuracy:.0f}%)\n')
    return accuracy

# We need to write the function to evaluate individual models and compute standard deviation
def evaluate_individual(models, device, test_loader):
    accuracies = []
    with torch.no_grad():
        for model in models:
            total_correct = 0
            total_data = 0
            model.eval()
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = model(data, active_segments=list(range(len(model.segments))))
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_data += target.size(0)
            accuracy = 100. * total_correct / total_data
            accuracies.append(accuracy)
            print(f'Participant Model Accuracy: {accuracy:.2f}%')
    std_dev = np.std(accuracies)
    print(f'Standard Deviation of Accuracies: {std_dev:.2f}')
    return accuracies, std_dev

# Main block for execution
train_dataset, test_dataset = load_mnist_data()

num_seeds = 5
participant_accuracies = {i: [] for i in range(2, 11)}

for seed in range(num_seeds):
    set_seed(seed)

    for num_participants in range(2, 11):  # 2 to 10 participants
        print(f'\nTraining with {num_participants} participants...')
        partitioned_datasets = vertical_partition_rotating(train_dataset, num_participants)
        input_sizes = [784 // num_participants] * (num_participants - 1) + [784 - (num_participants - 1) * (784 // num_participants)]
        hidden_sizes = [10] * num_participants
        output_size = 10

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = [GlobalModel(input_sizes, hidden_sizes, output_size).to(device) for _ in range(num_participants)]
        optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

        for participant_id in range(num_participants):
            print(f"Training Participant {participant_id + 1} without federation")
            train_loader = DataLoader(partitioned_datasets[participant_id], batch_size=64, shuffle=True)
            for epoch in range(1, 26):  # Training for 25 epochs to mimic 5 Federated Rounds with 5 local epoch each
                train_without_federation(models[participant_id], device, train_loader, optimizers[participant_id], epoch, input_sizes, participant_id)
                # No sharing of gradient mechanism
        # Evaluation
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        print("Evaluating individual models from non-federated training:")
        accuracies, std_dev = evaluate_individual(models, device, test_loader)
        participant_accuracies[num_participants].append(np.mean(accuracies))

# Average accuracy for each number of participants
avg_accuracies = {num_participants: np.mean(accs) for num_participants, accs in participant_accuracies.items()}

"""
# ResultPlot
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), [avg_accuracies[i] for i in range(2, 11)], marker='o', linestyle='-', color='b')
plt.xlabel('Number of Participants')
plt.ylabel('Accuracy (%)')
plt.title('Average Accuracy vs. Number of Participants (Without Gradient Exchange)')
plt.grid(True)
plt.show()
"""
