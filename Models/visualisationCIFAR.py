import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from CIFAR2 import QuantumCNN  # Import the QuantumCNN model class
#
# # Check if CUDA is available and set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# # Define test dataset transformations
# transform_test = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# # Load CIFAR-10 test dataset
# test_data = datasets.CIFAR10(
#     root='data',
#     train=False,
#     transform=transform_test,
#     download=True
# )
#
# # Create DataLoader for validation set
# val_loader = DataLoader(test_data, batch_size=64, shuffle=False)
#
# # Load the trained model
# model = QuantumCNN()
# model.load_state_dict(torch.load("output_results/cifar2_model.pth", map_location=device))
# model.to(device)
# model.eval()  # Set to evaluation mode
#
# print("Model loaded successfully!")
#
# # CIFAR-10 class names
# classes = [
#     "airplane", "automobile", "bird", "cat", "deer",
#     "dog", "frog", "horse", "ship", "truck"
# ]
#
# # Function to visualize random validation images and their predicted labels
# def visualize_predictions(model, val_loader, num_images=20):
#     # Get a batch of shuffled validation images
#     shuffled_loader = DataLoader(val_loader.dataset, batch_size=num_images, shuffle=True)
#     data_iter = iter(shuffled_loader)
#     images, labels = next(data_iter)
#
#     # Move images to the device used for model inference (CPU or GPU)
#     images = images.to(device)
#     labels = labels.to(device)
#
#     # Get predictions from the model
#     outputs = model(images)
#     _, predicted = torch.max(outputs, 1)
#
#     # Plot the images and predictions
#     fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # 4 rows, 5 columns
#     images = images.cpu()  # Move images back to CPU for visualization
#
#     for idx, ax in enumerate(axes.flat):
#         img = images[idx].numpy().transpose((1, 2, 0))  # Convert tensor to image format
#         ax.imshow(img)
#         ax.set_title(f"True: {classes[labels[idx]]}\nPred: {classes[predicted[idx]]}")
#         ax.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
# # Visualize 20 random predictions from the validation set
# visualize_predictions(model, val_loader, num_images=20)

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. SETUP - DEVICE CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. DATA LOADING (NO TRANSFORMS THAT WERE USED IN TRAINING)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Must match training
])

test_data = datasets.CIFAR10(
    root='data',
    train=False,
    transform=transform,
    download=True
)

val_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 3. MODEL LOADING
model = QuantumCNN().to(device)
model.load_state_dict(torch.load("output_results/cifar2_model.pth", map_location=device))
model.eval()  # Crucial for prediction mode


# 4. PREDICTION FUNCTION
def visualize_predictions(model, loader, num_images=20):
    # Get random batch
    data_iter = iter(DataLoader(loader.dataset, batch_size=num_images, shuffle=True))
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    # Prediction (no gradients)
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # Denormalize for display
    images = images.cpu()
    images = images * 0.5 + 0.5  # Reverse normalization

    # Plot
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for idx, ax in enumerate(axes.flat):
        if idx >= num_images:
            break
        img = images[idx].numpy().transpose((1, 2, 0))
        ax.imshow(img)
        ax.set_title(f"True: {classes[labels[idx]]}\nPred: {classes[preds[idx]]}",
                     color='green' if labels[idx] == preds[idx] else 'red')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# 5. CLASS NAMES
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 6. EXECUTE VISUALIZATION
visualize_predictions(model, val_loader)

#
# import matplotlib.pyplot as plt
# import torch
#
# import torch
# import torchvision
# from torchvision import transforms, datasets
# from torchvision.transforms import ToTensor
# import torch.optim as optim
# import torch.nn as nn
# import pennylane as qml
# from pennylane import numpy as np
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, random_split, Subset
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# import random
# import os
#
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
#
# # Download and load MNIST dataset
# train_data = datasets.MNIST(root = 'data',
#     train = True,
#     transform = ToTensor(),
#     download = True,
# )
# test_data = datasets.MNIST(
#     root = 'data',
#     train = False,
#     transform = ToTensor()
# )
#
# # Number of samples in the training dataset
# num_train_samples = len(train_data)
# print(f"Number of training samples: {num_train_samples}")
#
# # Number of samples in the test dataset
# num_test_samples = len(test_data)
# print(f"Number of test samples: {num_test_samples}")
#
# # Set a fixed seed for all random number generators
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
#
# # For reproducibility on GPU (optional, but might slightly affect performance)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#
# # Quantum device setup
# num_qubits = 5  # Set the number of qubits
# num_layers = 3  # Set the number of rotation layers
# L = 3  # Number of layers in the entangling layer
#
# quantum_device = qml.device("default.qubit", wires=num_qubits)
#
# # Quantum node for a single PQC
# @qml.qnode(quantum_device)
# def quantum_circuit(inputs, weights, entangling_weights):
#
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
#
# # Define shapes for weights
# param_shapes = {"weights": (num_layers, num_qubits, 3), "entangling_weights": (L, num_qubits)}
#
# # Neural network architecture with quantum layers
# class QuantumNet(nn.Module):
#     def __init__(self):
#         super(QuantumNet, self).__init__()
#
#         # First convolutional layer: 1 input channel, 16 output channels, 5x5 kernel
#         self.conv_layer1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
#
#         # Batch normalization with output from cn1
#         self.batch_norm1 = nn.BatchNorm2d(16)
#
#         # Max pooling layer with a 2x2 window
#         self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # Second convolutional layer: 16 input channels, 32 output channels, 5x5 kernel
#         self.conv_layer2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
#
#         # Batch normalization with output from cn2
#         self.batch_norm2 = nn.BatchNorm2d(32)
#
#         # Quantum layers using PennyLane
#         self.quantum_layer1 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)
#         self.quantum_layer2 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)
#         self.quantum_layer3 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)
#         self.quantum_layer4 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)
#         self.quantum_layer5 = qml.qnn.TorchLayer(quantum_circuit, param_shapes)
#
#         # Fully connected layers for classification
#         self.fc_layer1 = nn.Linear(32 * 7 * 7, 120)
#         self.fc_layer2 = nn.Linear(120, 20)
#         self.fc_output = nn.Linear(25, 10)  # 10 classes
#
#         # Dropout
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#
#         # Forward pass through the first convolutional layer
#         x = self.conv_layer1(x)
#         x = F.relu(x)
#         x = self.batch_norm1(x)
#         x = self.max_pool(x)
#
#         # Forward pass through the second convolutional layer
#         x = self.conv_layer2(x)
#         x = F.relu(x)
#         x = self.batch_norm2(x)
#         x = self.max_pool(x)
#
#         # Flatten the output from the convolutional layers
#         x = x.view(-1, 32 * 7 * 7)
#
#         # Forward pass through the first fully connected layer
#         x = self.fc_layer1(x)
#         x = F.relu(x)
#         x = self.dropout(x)  # Apply dropout
#
#         # Forward pass through the second fully connected layer
#         x = self.fc_layer2(x)
#         x = F.relu(x)
#
#         # Split the output into four parts for the quantum layers
#         x_1, x_2, x_3, x_4, x_5 = torch.split(x, 4, dim=1)
#
#         # Forward pass through each quantum layer
#         x_1 = self.quantum_layer1(x_1)
#         x_2 = self.quantum_layer2(x_2)
#         x_3 = self.quantum_layer3(x_3)
#         x_4 = self.quantum_layer4(x_4)
#         x_5 = self.quantum_layer5(x_5)
#
#         # Concatenate the outputs from the quantum layers
#         x = torch.cat([x_1, x_2, x_3, x_4, x_5], dim=1)
#
#         # Forward pass through the output layer
#         x = self.fc_output(x)
#         return x
#
# # Initialize the neural network model
# model = QuantumNet()
#
# # Create data loaders for training and validation
# train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# val_loader = DataLoader(test_data, batch_size=64, shuffle=False)
#
#
# # Function to visualize random validation images and their predicted labels
# def visualize_predictions(model, val_loader):
#     # Create a shuffled loader with a small batch size for visualization
#     shuffled_loader = DataLoader(val_loader.dataset, batch_size=5, shuffle=True)
#     data_iter = iter(shuffled_loader)
#     images, labels = next(data_iter)
#
#     # Get model predictions
#     with torch.no_grad():
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#
#     # Create figure with subplots
#     fig, axes = plt.subplots(1, 5, figsize=(15, 5))
#     for idx in range(5):
#         ax = axes[idx]
#         ax.imshow(images[idx].numpy().squeeze(), cmap='gray')
#         ax.set_title(f"True: {labels[idx].item()}\nPred: {predicted[idx].item()}")
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
#
# # Load your trained model weights
# model.load_state_dict(torch.load('"output_results/mnist2_model.pth", map_location=device'))  # Replace with your actual .pth file path
# model.eval()  # Set the model to evaluation mode
#
# # Visualize predictions with the trained model
# visualize_predictions(model, val_loader)
#
# # Visualize random predictions from the validation set
# visualize_predictions(model, val_loader)