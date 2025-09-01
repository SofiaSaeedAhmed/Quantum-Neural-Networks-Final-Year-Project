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
import time


# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single-channel images
    # transforms.Resize((244, 244)),                  # Resize to match the input size
    transforms.RandomRotation(degrees=10),         # Randomly rotate up to 10 degrees
    transforms.RandomHorizontalFlip(p=0.5),        # Randomly flip horizontally with 50% probability
    transforms.ToTensor(),                        # Convert to tensor
])


# Load Medical MNIST dataset
data_dir = 'medical_mnist'  # Ensure this folder contains all 6 class folders
dataset = datasets.ImageFolder(root=data_dir, transform=transform)


# Split into train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

# Class labels
classes = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]


train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

#
# # Quantum device setup
# num_qubits = 5  # Set the number of qubits
# num_layers = 3  # Set the number of parallel circuits
# L = 3  # Number of layers in the entangling layer
#
# quantum_device = qml.device("default.qubit", wires=num_qubits)
#
# # Quantum node for a single PQC
# @qml.qnode(quantum_device)
# def quantum_circuit(inputs, weights, entangling_weights):
#     # Angle Embedding for input features
#     qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='X')
#
#     # Apply RZ-Y-RZ rotations with initial weights
#     for i in range(num_qubits):
#         qml.RZ(weights[0, i, 0], wires=i)
#         qml.RY(weights[0, i, 1], wires=i)
#         qml.RZ(weights[0, i, 2], wires=i)
#
#     # Entangling layer using entangling weights
#     qml.BasicEntanglerLayers(entangling_weights, wires=range(num_qubits))
#
#     # Measurement: expectation values of Pauli-Y for each qubit
#     return [qml.expval(qml.PauliY(i)) for i in range(num_qubits)]
#
# # Define shapes for weights
# param_shapes = {"weights": (num_layers, num_qubits, 3), "entangling_weights": (L, num_qubits)}
#
# # Neural Network architecture with quantum layers
# class QuantumCNN(nn.Module):
#     def __init__(self):
#         super(QuantumCNN, self).__init__()
#
#         # Convolutional layers with Batch Normalization
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         # self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
#         # self.bn3 = nn.BatchNorm2d(128)
#         # self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=1)
#         # self.bn4 = nn.BatchNorm2d(256)
#
#         # Pooling layer
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # Fully connected layers
#         self.fc1 = nn.Linear(32 * 16 * 16, 128)
#         self.bn_fc1 = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(128, 64)
#         self.bn_fc2 = nn.BatchNorm1d(64)
#         self.fc3 = nn.Linear(64, 20)
#         self.bn_fc3 = nn.BatchNorm1d(20)
#
#         self.fc_output = nn.Linear(20, 6)
#
#         # Quantum layers using PennyLane
#         self.quantum_layer1 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)
#         self.quantum_layer2 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)
#         self.quantum_layer3 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)
#         self.quantum_layer4 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)
#
#         # Dropout
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         # Convolutional layers
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#
#         # Flatten with correct dimensions
#         x = x.view(x.size(0), -1)  # This should give [batch_size, 8192]
#
#         # Fully connected layers
#         x = F.relu(self.bn_fc1(self.fc1(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn_fc2(self.fc2(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn_fc3(self.fc3(x)))
#
#         # Quantum layers
#         x_1, x_2, x_3, x_4 = torch.split(x, 5, dim=1)
#         x_1 = self.quantum_layer1(x_1)
#         x_2 = self.quantum_layer2(x_2)
#         x_3 = self.quantum_layer3(x_3)
#         x_4 = self.quantum_layer4(x_4)
#
#         # Final output
#         x = torch.cat([x_1, x_2, x_3, x_4], dim=1)
#         x = self.fc_output(x)
#         return F.log_softmax(x, dim=1)

# Classical Neural Network architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=1)
        # self.bn4 = nn.BatchNorm2d(256)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 20)
        self.bn_fc3 = nn.BatchNorm1d(20)

        self.fc_output = nn.Linear(20, 6)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Flatten with correct dimensions
        x = x.view(x.size(0), -1)  # This should give [batch_size, 8192]

        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc3(self.fc3(x)))

        x = self.fc_output(x)
        return F.log_softmax(x, dim=1)

# Load the trained model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("output_results/medicalsimple_model.pth", map_location=device))
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
    print(f"Run {run+1}/{num_runs}: Test Accuracy = {test_accuracy:.2f}%")

# Compute average and standard deviation
mean_accuracy = np.mean(test_accuracies)
std_accuracy = np.std(test_accuracies)

# Print the results with clear formatting
print("\n" + "="*50)
print(f"Average Test Accuracy over {num_runs} runs: {mean_accuracy:.2f}%")
print(f"Standard Deviation: {std_accuracy:.2f}%")
print("="*50 + "\n")

# Visualization of test accuracies across runs
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_runs+1), test_accuracies, 'b-', label='Test Accuracy')
plt.axhline(y=mean_accuracy, color='r', linestyle='--', label=f'Mean: {mean_accuracy:.2f}%')
plt.fill_between(range(1, num_runs+1),
                 mean_accuracy-std_accuracy,
                 mean_accuracy+std_accuracy,
                 color='r', alpha=0.1, label=f'±1 std dev')
plt.xlabel('Run Number')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Across Multiple Runs')
plt.legend()
plt.grid(True)
plt.show()

# Function to unnormalize and convert tensor to NumPy image
def unnormalize(img):
    img = img * 0.5 + 0.5  # Reverse normalization (assuming mean=0.5, std=0.5)
    np_img = img.cpu().numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    return np_img


# Get a batch of test data
model.eval()
data_iter = iter(test_loader)
images, true_labels = next(data_iter)  # Get one batch (64 images)

# Move data to device (GPU if available)
images = images.to(device)
true_labels = true_labels.to(device)

# Get model predictions
with torch.no_grad():
    outputs = model(images)
_, predicted_labels = torch.max(outputs, 1)

# Display the first 10 images with predictions
num_images = 10
fig, axes = plt.subplots(2, 5, figsize=(15, 8))

for idx, ax in enumerate(axes.flat[:num_images]):
    img = unnormalize(images[idx])  # Unnormalize and convert to NumPy format

    # Determine if prediction was correct
    is_correct = predicted_labels[idx] == true_labels[idx]
    title_color = 'green' if is_correct else 'red'

    # Plot image
    ax.imshow(img.squeeze(), cmap='gray')  # Remove channel dim for grayscale
    ax.set_title(
        f"True: {classes[true_labels[idx]]}\nPred: {classes[predicted_labels[idx]]}",
        color=title_color,
        fontsize=10
    )
    ax.axis("off")

plt.tight_layout()
plt.show()


