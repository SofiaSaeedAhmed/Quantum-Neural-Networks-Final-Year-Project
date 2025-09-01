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
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print("All imports loaded successfully.")
print(f"CUDA Available: {use_cuda}, Device: {device}")

# -----------------------------------------------------------------------------------------

# Quantum device setup
num_qubits = 6  # Set the number of qubits
num_layers = 5  # Set the number of layers
L = 3  # Number of layers in the entangling layer
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

    # Entangling layer using entangling weights
    qml.BasicEntanglerLayers(entangling_weights, wires=range(num_qubits))

    # Measurement: expectation values of Pauli-Y for each qubit
    return [qml.expval(qml.PauliY(i)) for i in range(num_qubits)]

# Define shapes for weights
param_shapes = {"weights": (num_layers, num_qubits, 3), "entangling_weights": (L, num_qubits)}

# Example input and weight parameters
inputs = np.random.random(num_qubits)  # Random inputs
weights = np.random.random((num_layers, num_qubits, 3))  # Weights for rotation layers
entangling_weights = np.random.random((L, num_qubits))  # Weights for entangling layers

# Draw the circuit
fig, ax = qml.draw_mpl(quantum_circuit)(inputs, weights, entangling_weights)
plt.show()
