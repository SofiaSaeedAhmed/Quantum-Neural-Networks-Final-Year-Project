import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import os
import time
import glob
from PIL import Image
import matplotlib.pyplot as plt

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Define dataset path
data_dir = r"C:\Users\sofia\Desktop\Year 3\FYP\Models\.venv\temp_all_frames"  # Change path accordingly

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Custom Dataset Class
class UCF101Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for img_path in glob.glob(os.path.join(cls_dir, "*.jpg")):
                self.samples.append((img_path, self.class_to_idx[cls]))

        # Shuffle dataset
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Load full dataset
full_dataset = UCF101Dataset(data_dir, transform=transform)

# Split dataset (80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Set a fixed seed for all random number generators
seed = 42
torch.manual_seed(seed)
random.seed(seed)

# For reproducibility on GPU (optional, but might slightly affect performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define the CNN model (classical only)
class ClassicalCNN(nn.Module):
    def __init__(self):
        super(ClassicalCNN, self).__init__()
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 20)
        self.fc_output = nn.Linear(20, 20)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)  # Apply dropout

        # Forward pass through the second fully connected layer
        x = self.fc2(x)
        x = F.relu(x)

        # Forward pass through the output layer
        x = self.fc_output(x)
        return x

if __name__ == "__main__":
    # Initialize the CNN model
    model = ClassicalCNN()
    model.to(device)  # Move model to GPU if available
    print(model)

    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Training settings
    epochs = 10
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)  # Create directory for results
    log_file = os.path.join(output_dir, "cnnUCf_output.txt")

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

        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = loss_function(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model parameters

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
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = val_correct / val_total
        val_acc_history.append(val_accuracy)
        print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Write to file
        with open(log_file, "a") as f:
            f.write(f"{epoch + 1},{avg_train_loss:.4f},{train_accuracy:.4f},{avg_val_loss:.4f},{val_accuracy:.4f}\n")

    print("Training Completed.")

    # Measure the training end time
    end_time = time.time()

    # Calculate the total time taken
    training_time = end_time - start_time
    print(f"Training Completed in {training_time:.2f} seconds")

    # Save the trained model
    model_path = os.path.join(output_dir, "cnnUCF_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print(f"Training log saved to {log_file}")
    print('Training Completed')

    # Final evaluation
    correct_predictions = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    final_accuracy = correct_predictions / total_samples
    print(f'Final Test Accuracy: {final_accuracy * 100:.2f}%')

    def visualize_predictions(model, test_loader, class_names, num_images=20):
        """Visualizes predictions from the test set."""
        model.eval()
        shuffled_loader = DataLoader(test_loader.dataset, batch_size=num_images, shuffle=True)
        data_iter = iter(shuffled_loader)
        images, labels = next(data_iter)

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # 4 rows, 5 columns
        images = images.cpu()  # Move images back to CPU for visualization

        for idx, ax in enumerate(axes.flat):
            img = images[idx].numpy().transpose((1, 2, 0))  # Convert tensor to image format
            img = (img * 0.5) + 0.5  # Unnormalize
            ax.imshow(img)
            ax.set_title(f"True: {class_names[labels[idx]]}\nPred: {class_names[predicted[idx]]}", fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    # After training, visualize predictions
    visualize_predictions(model, test_loader, full_dataset.classes, num_images=20)