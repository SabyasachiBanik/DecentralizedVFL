import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# Reproducibility with Seeding
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
        full_images = [torch.zeros_like(image) for _ in range(num_participants)]  # Each image is initialized as a zero tensor with the same shape as the original image, effectively zero-padding all features initially. Only the features corresponding to each participant are filled with actual data, while the rest remain as zeros.
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



class ParticipantModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_participants):
        super(ParticipantModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size * num_participants, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        hidden_output = self.hidden_layer(self.input_layer(x))
        hidden_output = self.dropout(hidden_output)
        return hidden_output

    def predict(self, combined_hidden_outputs):
        output = self.output_layer(combined_hidden_outputs)
        return output

# Function to generate hidden outputs for each participant
def generate_hidden_outputs(models, data, device):
    hidden_outputs = []
    for model in models:
        hidden_output = model(data.to(device))
        hidden_outputs.append(hidden_output)
    return hidden_outputs

def train(models, device, train_loader, optimizers, epoch, input_size):
    start_time = time.time()
    for model in models:
        model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start_time = time.time()
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)  # We flatten the image into a single vector, preserving the zero values for non-responsible features

        hidden_outputs = generate_hidden_outputs(models, data, device)
        combined_hidden_outputs = torch.cat(hidden_outputs, dim=1)

        # For Backpropagation loss sharing mechanism, we first append the loss from each model
        losses = []
        for i, model in enumerate(models):
            output = model.predict(combined_hidden_outputs)
            loss = nn.CrossEntropyLoss()(output, target)
            losses.append(loss)

        # Next we share the backpropagation loss
        total_loss = sum(losses)
        total_loss.backward()

        # Next up, need to update the model with these shared gradients
        for i, model in enumerate(models):
            optimizer = optimizers[i]
            optimizer.step()
            optimizer.zero_grad()

# Function to safely initialize or get gradients
def safe_gradient(parameter):
    if parameter.grad is not None:
        return parameter.grad.data
    else:
        return torch.zeros_like(parameter.data)

# Selective gradient exchange function
def selective_exchange_gradients(models, hidden_size):
    num_models = len(models)
    total_hidden_size = hidden_size * num_models
    param_indices = [0]
    cumulative_index = 0
    for i in range(num_models):
        cumulative_index += total_hidden_size
        param_indices.append(cumulative_index)

    for seg in range(num_models):
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

def evaluate(models, device, test_loader):
    for model in models:
        model.eval()

    with torch.no_grad():
        test_loss = 0
        correct = 0
        all_preds = []
        all_targets = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)

            hidden_outputs = generate_hidden_outputs(models, data, device)
            combined_hidden_outputs = torch.cat(hidden_outputs, dim=1)
            outputs = torch.zeros(data.size(0), 10, device=device)
            for model in models:
                output = model.predict(combined_hidden_outputs)
                outputs += output
            outputs /= len(models)

            test_loss += nn.CrossEntropyLoss(reduction='sum')(outputs, target).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.view(-1).cpu().numpy())

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        #print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')
        return accuracy

# Main block for execution
train_dataset, test_dataset = load_fashion_mnist_data()

num_seeds = 5
participant_accuracies = {i: [] for i in range(2, 11)}

for seed in range(num_seeds):
    set_seed(seed)

    for num_participants in range(2, 11):  # 2 to 11 participants
        partitioned_datasets = vertical_partition_rotating(train_dataset, num_participants)
        input_size = 784  # Each participant has to handle the entire input size with zero-padding
        hidden_size = 10  
        output_size = 10

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        participant_models = [ParticipantModel(input_size, hidden_size, output_size, num_participants).to(device) for _ in range(num_participants)]

        optimizers = {i: optim.Adam(model.parameters(), lr=0.001) for i, model in enumerate(participant_models)}

        federated_rounds = 5
        epochs_per_round = 5

        for federated_round in range(federated_rounds):
            for participant_id in range(num_participants):
                train_loader = DataLoader(partitioned_datasets[participant_id], batch_size=64, shuffle=True, num_workers=4)
                for epoch in range(1, epochs_per_round + 1):
                    train(participant_models, device, train_loader, optimizers, epoch, input_size)
            selective_exchange_gradients(participant_models, hidden_size)
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        accuracy = evaluate(participant_models, device, test_loader)
        participant_accuracies[num_participants].append(accuracy)
        print(f"Seed: {seed}, Number of Participants: {num_participants}, Accuracy: {accuracy:.2f}%")

# Calculate the average accuracy for each number of participants
avg_accuracies = {num_participants: np.mean(accs) for num_participants, accs in participant_accuracies.items()}


"""
# ResultsPlot # We plot all accuracies in a separate file - Accuracy Plots

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), [avg_accuracies[i] for i in range(2, 11)], marker='o', linestyle='-', color='b')
plt.xlabel('Number of Participants')
plt.ylabel('Accuracy (%)')
plt.title('Average Accuracy vs. Number of Participants')
plt.grid(True)
plt.show()

"""