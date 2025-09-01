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
import glob
from PIL import Image
import cv2

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Dataset Path
video_data_dir = "C:/Users/sofia/Desktop/Year 3/FYP/Models/.venv/UCF20"  # CHANGE PATH

# Video Preprocessing
# Data Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# **Custom Dataset for Videos**
class UCF10VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, frames_per_video=16):
        self.data_dir = data_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect video paths
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            if os.path.isdir(cls_dir):
                for video_file in os.listdir(cls_dir):
                    if video_file.endswith((".mp4", ".avi")):
                        self.samples.append((os.path.join(cls_dir, video_file), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_video_frames(video_path)
        return frames, label

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)  # Select evenly spaced frames

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frame = Image.fromarray(frame)  # Convert to PIL Image
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)

        cap.release()

        # Pad with last frame if video has fewer frames
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1])

        frames = torch.stack(frames)  # Shape: (T, C, H, W)
        return frames


# Load Dataset
frames_per_video = 16  # Use 16 frames per video
dataset = UCF10VideoDataset(video_data_dir, transform=transform, frames_per_video=frames_per_video)

# Split Dataset (80% Train, 20% Test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Set a fixed seed for all random number generators
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# For reproducibility on GPU (optional, but might slightly affect performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# # Define CNN model
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = x.view(x.size(0), -1)  # Flatten before passing to LSTM
#         return x
#
# # Define LSTM model with CNN integrated
# class CNN_LSTM(nn.Module):
#     def __init__(self, num_classes=20, hidden_dim=128, num_layers=2, dropout_rate=0.5):
#         super(CNN_LSTM, self).__init__()
#         self.cnn = SimpleCNN()  # feature extraction through CNN
#
#         # Dynamically calculate CNN output size
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, 3, 64, 64)
#             self.cnn_output_size = self.cnn(dummy_input).shape[1]
#
#         # LSTM Model initialize
#         self.lstm = nn.LSTM(input_size=self.cnn_output_size,
#                             hidden_size=hidden_dim,
#                             num_layers=num_layers,
#                             batch_first=True,
#                             dropout=dropout_rate)
#
#         self.bn_lstm = nn.BatchNorm1d(hidden_dim)
#         self.dropout = nn.Dropout(dropout_rate)
#
#         # Final Classification Layer
#         self.fc = nn.Linear(hidden_dim, num_classes)
#
#     def forward(self, x):
#         batch_size, seq_length, C, H, W = x.shape
#         cnn_features = []
#
#         for t in range(seq_length):
#             frame = x[:, t, :, :, :]
#             frame_features = self.cnn(frame)
#             cnn_features.append(frame_features)
#
#         cnn_features = torch.stack(cnn_features, dim=1)
#
#         lstm_out, _ = self.lstm(cnn_features)
#         lstm_out = self.bn_lstm(lstm_out[:, -1, :])
#         lstm_out = self.dropout(lstm_out)
#
#         final_out = self.fc(lstm_out)
#         return final_out

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 20)
        self.bn_fc2 = nn.BatchNorm1d(20)

        self.fc_output = nn.Linear(20, 20)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))

        # Removed quantum layers and kept the classical processing
        x = self.fc_output(x)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=20, hidden_dim=256, num_layers=2):
        super(CNN_LSTM, self).__init__()
        self.cnn = SimpleCNN()

        with torch.no_grad():
            self.cnn.eval()  # Set CNN to eval mode (disables BatchNorm issues)
            dummy_input = torch.zeros(2, 3, 64, 64)  # Use at least batch size = 2
            self.cnn_output_size = self.cnn(dummy_input).shape[1]
            self.cnn.train()  # Set CNN back to training mode

        self.lstm = nn.LSTM(input_size=self.cnn_output_size,
                           hidden_size=hidden_dim,
                           num_layers=num_layers,
                           batch_first=True)
        self.bn_lstm = nn.BatchNorm1d(hidden_dim)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_length, C, H, W = x.shape
        cnn_features = []

        for t in range(seq_length):
            frame = x[:, t, :, :, :]
            frame_features = self.cnn(frame)
            cnn_features.append(frame_features)

        cnn_features = torch.stack(cnn_features, dim=1)

        lstm_out, _ = self.lstm(cnn_features)
        lstm_out = self.bn_lstm(lstm_out[:, -1, :])

        final_out = self.fc(lstm_out)
        return final_out

        # Visualization function
    def visualize_results(show_correct=True, show_incorrect=True, num_images=10):
        classes = [
            "Basketball", "Biking", "CliffDiving", "CricketBowling", "Diving",
            "Fencing", "GolfSwing", "HorseRiding", "Kayaking", "PullUps",
            "PushUps", "RockClimbingIndoor", "RopeClimbing", "Rowing", "SkateBoarding",
            "Surfing", "Swing", "TennisSwing", "ThrowDiscus", "VolleyballSpiking"
        ]

        def unnormalize(img_tensor):
            img = img_tensor * 0.5 + 0.5
            return img.cpu().permute(1, 2, 0)

        # Collect samples
        samples = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for i in range(len(inputs)):
                    is_correct = preds[i] == labels[i]
                    if (show_correct and is_correct) or (show_incorrect and not is_correct):
                        samples.append((inputs[i], labels[i], preds[i]))
                    if len(samples) >= num_images:
                        break
                if len(samples) >= num_images:
                    break

        # Visualize
        rows = int(np.ceil(num_images / 5))
        fig, axes = plt.subplots(rows, 5, figsize=(20, rows * 4))
        fig.suptitle('Model Predictions (Green=Correct, Red=Incorrect)', fontsize=16)

        for idx, ax in enumerate(axes.flat if rows > 1 else axes):
            if idx >= len(samples):
                ax.axis('off')
                continue

            img, true_label, pred_label = samples[idx]
            img = unnormalize(img)
            is_correct = true_label == pred_label

            ax.imshow(img)
            ax.set_title(
                f"True: {classes[true_label]}\nPred: {classes[pred_label]}",
                color='green' if is_correct else 'red',
                fontsize=10
            )
            ax.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Initialize the LSTM model
    model = CNN_LSTM(num_classes=20).to(device)

    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Training settings
    epochs = 10
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    output_dir = "output_results"
    os.makedirs(output_dir,
                exist_ok=True)  # Create directory for results -----------------------------------------------
    log_file = os.path.join(output_dir, "lstmCNN_ucf.txt")

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
            optimizer.zero_grad()  # Zero the parameter gradients
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
    model_path = os.path.join(output_dir, "lstmCNN_ucf.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print(f"Training log saved to {log_file}")
    print('Training Completed')

    # Measure the training end time
    end_time = time.time()

    # Calculate the total time taken
    training_time = end_time - start_time
    print(f"Training Completed in {training_time:.2f} seconds")

    # Evaluate the model on validation data
    correct_predictions = 0
    total_samples = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    final_accuracy = correct_predictions / total_samples
    print(f'Final Test Accuracy: {final_accuracy * 100:.2f}%')


    # Run evaluations and visualizations
    evaluate_model()
    visualize_results(show_correct=True, show_incorrect=True)  # Show both
    visualize_results(show_correct=False, show_incorrect=True)  # Show only misclassifications




