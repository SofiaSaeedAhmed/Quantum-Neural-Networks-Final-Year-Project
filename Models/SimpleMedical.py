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
print(f"CUDA Available: {use_cuda}, Device: {device}")

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

# Collect one image per class
class_images = {}
class_labels = {}

for img, label in dataset:
    if label not in class_images:
        class_images[label] = img
        class_labels[label] = classes[label]
    if len(class_images) == len(classes):
        break

# Plot the images
fig, axes = plt.subplots(1, 6, figsize=(12, 4))
for idx, label in enumerate(sorted(class_images.keys())):
    img = class_images[label].numpy().transpose((1, 2, 0))  # Convert to HWC
    axes[idx].imshow(img.squeeze(), cmap='gray')
    axes[idx].set_title(class_labels[label], fontsize=10)
    axes[idx].axis('off')

# plt.tight_layout()
# plt.show()


# Number of samples
num_train_samples = len(train_data)
print(f"Number of training samples: {num_train_samples}")

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

# Classical Neural Network architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

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

if __name__ == "__main__":
    # Initialize the CNN model
    model = SimpleCNN()
    print(model)

    # Define loss function and optimizer
    loss_function = nn.NLLLoss()

    # Change the learning rate to improve the accuracy
    # adam_optimizer = optim.Adam(model.parameters(), lr=0.001)

    classical_params = [p for name, p in model.named_parameters() if not "quantum_layer" in name]

    classical_optimizer = optim.Adam(classical_params, lr=0.003)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Training settings
    epochs = 10
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    best_val_accuracy = 0.0
    best_model_state = None

    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)  # Create directory for results -----------------------------------------------
    log_file = os.path.join(output_dir, "medicalsimple_output.txt")

    # Save the epoch data to a file
    with open(log_file, "w") as f:
        f.write("Epoch,Train Loss,Train Accuracy,Val Loss,Val Accuracy\n")

    # Measure the training start time
    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        model.train()  # Set the model to training mode
        for inputs, labels in train_loader:
            # adam_optimizer.zero_grad()  # Zero the parameter gradients
            classical_optimizer.zero_grad()

            outputs = model(inputs)  # Forward pass
            loss = loss_function(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass

            # Step optimizers
            # adam_optimizer.step()  # Optimize the model parameters

            classical_optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_accuracy)
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation
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

        # Write to file ----------------------------------------------------------------------------------
        with open(log_file, "a") as f:
            f.write(f"{epoch + 1},{avg_train_loss:.4f},{train_accuracy:.4f},{avg_val_loss:.4f},{val_accuracy:.4f}\n")

        # Checkpoint for the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()  # Save the best model state

    print('Training Completed')

    # Measure the training end time
    end_time = time.time()

    # Calculate the total time taken
    training_time = end_time - start_time
    print(f"Training Completed in {training_time:.2f} seconds")

    # Save the trained model
    model_path = os.path.join(output_dir, "medicalsimple_model.pth")       # --------------------------------------------------------
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print(f"Training log saved to {log_file}")
    print('Training Completed')

    # Load the best model state for final evaluation
    # if best_model_state is not None:
    #     model.load_state_dict(best_model_state)
    #     print(f'Best Validation Accuracy: {best_val_accuracy * 100:.2f}%')

    # Final evaluation on validation data
    correct_predictions = 0
    total_samples = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    final_accuracy = correct_predictions / total_samples
    print(f'Final Test Accuracy: {final_accuracy * 100:.2f}%')


