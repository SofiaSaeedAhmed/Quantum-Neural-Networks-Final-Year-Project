import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Download and load MNIST dataset
train_data = datasets.MNIST(root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,
)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor()
)

# Number of samples in the training dataset
num_train_samples = len(train_data)
print(f"Number of training samples: {num_train_samples}")

# Number of samples in the test dataset
num_test_samples = len(test_data)
print(f"Number of test samples: {num_test_samples}")


# Create data loaders for training and validation
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Set a fixed seed for all random number generators
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# For reproducibility on GPU (optional, but might slightly affect performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Quantum device setup
num_qubits = 5  # Set the number of qubits
num_layers = 3
L = 3

quantum_device = qml.device("default.qubit", wires=num_qubits)

# Quantum node for a single PQC
@qml.qnode(quantum_device)
def quantum_circuit(inputs, weights, entangling_weights):
    # Angle Embedding for input features
    qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='X')

    # Apply RZ-Y-RZ rotations with initial weights

    for i in range(num_qubits):
        qml.RZ(weights[0, i, 0], wires=i)
        qml.RY(weights[0, i, 1], wires=i)
        qml.RZ(weights[0, i, 2], wires=i)

    # Apply entanglement using CNOT gates for this layer
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    # Wrap entanglement (last qubit connected to the first for a cycle)
    qml.CNOT(wires=[num_qubits - 1, 0])

    # Measurement: expectation values of Pauli-Y for each qubit
    return [qml.expval(qml.PauliY(i)) for i in range(num_qubits)]

# Define shapes for weights
param_shapes = {"weights": (num_layers, num_qubits, 3), "entangling_weights": (L, num_qubits)}

# Neural network architecture with quantum layers
class QuantumNet(nn.Module):
    def __init__(self):
        super(QuantumNet, self).__init__()

        # First convolutional layer: 1 input channel, 16 output channels, 5x5 kernel
        self.conv_layer1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)

        # Batch normalization with output from cn1
        self.batch_norm1 = nn.BatchNorm2d(16)

        # Max pooling layer with a 2x2 window
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer: 16 input channels, 32 output channels, 5x5 kernel
        self.conv_layer2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)

        # Batch normalization with output from cn2
        self.batch_norm2 = nn.BatchNorm2d(32)

        # Quantum layers using PennyLane
        self.quantum_layer1 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)
        self.quantum_layer2 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)
        self.quantum_layer3 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)
        self.quantum_layer4 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)
        self.quantum_layer5 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)

        # Fully connected layers for classification
        self.fc_layer1 = nn.Linear(32 * 7 * 7, 120)
        self.fc_layer2 = nn.Linear(120, 20)
        self.fc_output = nn.Linear(25, 10)  # 10 classes

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        # Forward pass through the first convolutional layer
        x = self.conv_layer1(x)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.max_pool(x)

        # Forward pass through the second convolutional layer
        x = self.conv_layer2(x)
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = self.max_pool(x)

        # Flatten the output from the convolutional layers
        x = x.view(-1, 32 * 7 * 7)

        # Forward pass through the first fully connected layer
        x = self.fc_layer1(x)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout

        # Forward pass through the second fully connected layer
        x = self.fc_layer2(x)
        x = F.relu(x)

        # Split the output into four parts for the quantum layers
        x_1, x_2, x_3, x_4, x_5 = torch.split(x, 4, dim=1)

        # Forward pass through each quantum layer
        x_1 = self.quantum_layer1(x_1)
        x_2 = self.quantum_layer2(x_2)
        x_3 = self.quantum_layer3(x_3)
        x_4 = self.quantum_layer4(x_4)
        x_5 = self.quantum_layer5(x_5)

        # Concatenate the outputs from the quantum layers
        x = torch.cat([x_1, x_2, x_3, x_4, x_5], dim=1)

        # Forward pass through the output layer
        x = self.fc_output(x)
        return x


# Load the trained model
model = QuantumNet().to(device)
model.load_state_dict(torch.load("output_results/mnist1b_model.pth", map_location=device))
model.eval()  # Set to evaluation mode

# Run test evaluation multiple times
num_runs = 30
test_accuracies = []

for run in range(num_runs):
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    # Compute test accuracy for this run
    test_accuracy = 100 * correct_predictions / total_samples
    test_accuracies.append(test_accuracy)
    print(f"Run {run + 1}/{num_runs}: Test Accuracy = {test_accuracy:.2f}%")

# Compute average and standard deviation
mean_accuracy = np.mean(test_accuracies)
std_accuracy = np.std(test_accuracies)

print("\n🔥 Final Test Accuracy Results 🔥")
print(f"Average Test Accuracy over {num_runs} runs: {mean_accuracy:.2f}%")
print(f"Standard Deviation: {std_accuracy:.2f}%")



# # Initialize the neural network model
# model = QuantumNet()
#
# # Function to visualize random validation images and their predicted labels
# def visualize_predictions(model, val_loader):
#     shuffled_loader = DataLoader(val_loader.dataset, batch_size=5, shuffle=True)
#     data_iter = iter(shuffled_loader)
#     images, labels = next(data_iter)
#
#     outputs = model(images)
#     _, predicted = torch.max(outputs, 1)
#
#     fig, axes = plt.subplots(1, 5, figsize=(15, 5))
#     for idx in range(5):
#         ax = axes[idx]
#         ax.imshow(images[idx].numpy().squeeze(), cmap='gray')
#         ax.set_title(f"True: {labels[idx].item()}\nPred: {predicted[idx].item()}")
#         ax.axis('off')
#     plt.show()
#
# # Visualize random predictions from the validation set
# visualize_predictions(model, val_loader)