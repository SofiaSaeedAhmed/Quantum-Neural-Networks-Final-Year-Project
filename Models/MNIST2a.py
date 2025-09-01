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
num_layers = 3  # Set the number of rotation layers
L = 3  # Number of layers in the entangling layer

quantum_device = qml.device("default.qubit", wires=num_qubits)

# Quantum node for a single PQC2
@qml.qnode(quantum_device)
def quantum_circuit(inputs, weights, entangling_weights):
    # Angle Embedding for input features
    qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='X')

    # Apply RZ-Y-RZ rotations with initial weights
    for i in range(num_qubits):
        qml.RZ(weights[0, i, 0], wires=i)
        qml.RY(weights[0, i, 1], wires=i)
        qml.RZ(weights[0, i, 2], wires=i)

    # Entangling layer using entangling weights
    qml.BasicEntanglerLayers(entangling_weights, wires=range(num_qubits))

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
        self.fc_layer2 = nn.Linear(120, 25)
        self.fc_output = nn.Linear(25, 10)  # 10 classes

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

        # Forward pass through the second fully connected layer
        x = self.fc_layer2(x)
        x = F.relu(x)

        # Split the output into four parts for the quantum layers
        x_1, x_2, x_3, x_4, x_5 = torch.split(x, 5, dim=1)

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


# Initialize the neural network model
model = QuantumNet()

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
adam_optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create data loaders for training and validation
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Training settings
epochs = 20
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)  # Create directory for results ---------------
log_file = os.path.join(output_dir, "mnist3a_output.txt")

# Save the epoch data to a file
with open(log_file, "w") as f:
    f.write("Epoch,Train Loss,Train Accuracy,Val Loss,Val Accuracy\n")

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    model.train()  # Set the model to training mode
    for inputs, labels in train_loader:
        adam_optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = loss_function(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        adam_optimizer.step()  # Optimize the model parameters

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = train_correct / train_total
    train_loss_history.append(avg_train_loss)
    train_acc_history.append(train_accuracy)
    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    val_loss_history.append(avg_val_loss)
    val_acc_history.append(val_accuracy)
    print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Write to file
    with open(log_file, "a") as f:
        f.write(f"{epoch + 1},{avg_train_loss:.4f},{train_accuracy:.4f},{avg_val_loss:.4f},{val_accuracy:.4f}\n")

# Save the trained model
model_path = os.path.join(output_dir, "mnist3a_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
print(f"Training log saved to {log_file}")
print('Training Completed')

# Evaluate the model on validation data
correct_predictions = 0
total_samples = 0
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
print(f'Validation Accuracy: {100 * correct_predictions / total_samples:.2f}%')
