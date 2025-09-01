import os
import shutil
import random

# Paths (Update these)
UCF101_VIDEO_PATH = r"C:\Users\sofia\Desktop\Year 3\FYP\Models\.venv\datasetucf\UCF101\UCF-101"  # Folder containing all UCF101 videos
SPLITS_PATH = r"C:\Users\sofia\Desktop\Year 3\FYP\Models\.venv\datasetucf\UCF101TrainTestSplits-RecognitionTask\ucfTrainTestlist" # Folder containing train/test splits
OUTPUT_PATH = r"C:\Users\sofia\Desktop\Year 3\FYP\Models\.venv\ucf101_processed"  # Where train/test/val folders will be created

# Create output directories
os.makedirs(os.path.join(OUTPUT_PATH, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "test"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "val"), exist_ok=True)

def read_video_list(file_path):
    """Reads a UCF101 train/test split file and returns a list of video names."""
    with open(file_path, "r") as f:
        videos = [line.strip().split()[0] for line in f.readlines()]
    return videos

# Read train and test video lists
train_videos = read_video_list(os.path.join(SPLITS_PATH, "trainlist01.txt"))
test_videos = read_video_list(os.path.join(SPLITS_PATH, "testlist01.txt"))

# Shuffle and select 10% of train videos for validation
random.shuffle(train_videos)
val_videos = train_videos[:len(train_videos) // 10]  # 10% for validation
train_videos = train_videos[len(train_videos) // 10:]  # Remaining for training

def move_videos(video_list, dest_folder):
    """Moves videos from the original dataset to train/test/val folders."""
    for video in video_list:
        class_name = video.split("/")[0]  # Extract class name
        src_path = os.path.join(UCF101_VIDEO_PATH, video)
        dest_class_path = os.path.join(dest_folder, class_name)

        os.makedirs(dest_class_path, exist_ok=True)  # Create class folder if needed

        if os.path.exists(src_path):  # Move only if the video exists
            shutil.move(src_path, dest_class_path)
        else:
            print(f"Warning: {src_path} not found!")

# Move videos into respective folders
move_videos(train_videos, os.path.join(OUTPUT_PATH, "train"))
move_videos(test_videos, os.path.join(OUTPUT_PATH, "test"))
move_videos(val_videos, os.path.join(OUTPUT_PATH, "val"))

print("Dataset successfully organized into train, test, and val folders!")
