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

def main():
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
            frame_indices = np.linspace(0, total_frames - 1, self.frames_per_video,
                                        dtype=int)  # Select evenly spaced frames

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

    # Load model
    model = CNN_LSTM(num_classes=20).to(device)
    model.load_state_dict(torch.load("output_results/lstmCNN_ucf.pth", map_location=device))
    model.eval()

    # Evaluation metrics
    def evaluate_model():
        num_runs = 30
        test_accuracies = []

        for run in range(num_runs):
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            test_accuracies.append(accuracy)
            print(f"Run {run + 1}/{num_runs}: Test Accuracy = {accuracy:.2f}%")

        mean_acc = np.mean(test_accuracies)
        std_acc = np.std(test_accuracies)

        print("\n" + "=" * 50)
        print(f"Average Test Accuracy over {num_runs} runs: {mean_acc:.2f}%")
        print(f"Standard Deviation: {std_acc:.2f}%")
        print("=" * 50 + "\n")

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_runs + 1), test_accuracies, 'b-', label='Test Accuracy')
        plt.axhline(y=mean_acc, color='r', linestyle='--', label=f'Mean: {mean_acc:.2f}%')
        plt.fill_between(range(1, num_runs + 1), mean_acc - std_acc, mean_acc + std_acc, color='r', alpha=0.1)
        plt.xlabel('Run Number')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy Across Multiple Runs')
        plt.legend()
        plt.grid(True)
        plt.show()

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

    # Run evaluations and visualizations
    evaluate_model()
    visualize_results(show_correct=True, show_incorrect=True)  # Show both
    visualize_results(show_correct=False, show_incorrect=True)  # Show only misclassifications


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()