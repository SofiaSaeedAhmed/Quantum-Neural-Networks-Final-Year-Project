#
#
# # CODE TO STORE FRAMES FROM 10 CLASSES IN TEMP FOLDER!!!! --------------------------------------------------!!!

import os
import cv2
import random
import shutil
#
# # Set dataset paths
# dataset_path = r"C:\Users\sofia\Desktop\Year 3\FYP\Models\.venv\UCF20"  # Input videos
# temp_all_frames_path = r"C:\Users\sofia\Desktop\Year 3\FYP\Models\.venv\temp_all_frames"  # Temporary storage
#
# # Ensure temp folder exists
# os.makedirs(temp_all_frames_path, exist_ok=True)
#
# def extract_frames(video_path, output_folder, fps=10):
#     """Extract frames from a video at specified FPS."""
#     os.makedirs(output_folder, exist_ok=True)  # Create class folder if not exists
#     cap = cv2.VideoCapture(video_path)
#     video_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = max(1, int(video_fps / fps))  # Ensure at least 1 frame per interval
#
#     count = 0
#     frame_list = []  # Store frame paths for shuffling
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         if count % frame_interval == 0:
#             frame_filename = os.path.join(output_folder, f"frame_{len(os.listdir(output_folder)):06d}.jpg")
#             cv2.imwrite(frame_filename, frame)
#             frame_list.append(frame_filename)
#
#         count += 1
#
#     cap.release()
#     return frame_list
#
#
# # # Step 1: Extract frames into their class folders inside temp_all_frames
# for class_name in os.listdir(dataset_path):
#     class_folder = os.path.join(dataset_path, class_name)
#     output_class_folder = os.path.join(temp_all_frames_path, class_name)  # Create subfolder for class
#
#     if os.path.isdir(class_folder):
#         os.makedirs(output_class_folder, exist_ok=True)  # Ensure class folder exists
#         for video in os.listdir(class_folder):
#             if video.endswith((".mp4", ".avi")):
#                 video_path = os.path.join(class_folder, video)
#                 extract_frames(video_path, output_class_folder, fps=10)
#
# print("Frame extraction complete! All frames are in temp_all_frames sorted by class.")

# CODE TO SHUFFLE!! --------------------------------------------------------------------------!!!!!!
#
# import os
# import random
#
# # Path to temp folder containing class subfolders
# temp_all_frames_path = r"C:\Users\sofia\Desktop\Year 3\FYP\Models\.venv\UCF20"
#
# # Shuffle frames in each class folder
# for class_name in os.listdir(temp_all_frames_path):
#     class_folder = os.path.join(temp_all_frames_path, class_name)
#
#     if os.path.isdir(class_folder):
#         frames = [f for f in os.listdir(class_folder) if f.endswith(".jpg")]
#         random.shuffle(frames)  # Shuffle the list of filenames
#
#         # **First Pass: Rename to temporary names**
#         for i, frame in enumerate(frames):
#             old_path = os.path.join(class_folder, frame)
#             temp_path = os.path.join(class_folder, f"temp_{i:06d}.jpg")
#             os.rename(old_path, temp_path)
#
#         # **Second Pass: Rename to final names**
#         for i, temp_frame in enumerate(sorted(os.listdir(class_folder))):
#             temp_path = os.path.join(class_folder, temp_frame)
#             new_path = os.path.join(class_folder, f"frame_{i:06d}.jpg")
#             os.rename(temp_path, new_path)
#
# print("Shuffling complete! Frames in each class folder are now randomized.")


# CODE TO STORE 60,000 (600 EACH) IN UCF_20 FOLDER!!! ---------------------------------------------------!!!!!!!!!
# import os
# import shutil
# import random
#
# # Paths
temp_all_frames_path = r"C:\Users\sofia\Desktop\Year 3\FYP\Models\.venv\temp_all_frames"
final_dataset_path = r"C:\Users\sofia\Desktop\Year 3\FYP\Models\.venv\ucf20_frames"
max_frames_per_class = 6000  # Limit per class


# Ensure final dataset directory exists
os.makedirs(final_dataset_path, exist_ok=True)

# Process each class folder
for class_name in os.listdir(temp_all_frames_path):
    class_folder = os.path.join(temp_all_frames_path, class_name)
    output_class_folder = os.path.join(final_dataset_path, class_name)

    if os.path.isdir(class_folder):
        os.makedirs(output_class_folder, exist_ok=True)  # Create output class folder

        # Get all frame filenames and shuffle
        frames = [f for f in os.listdir(class_folder) if f.endswith(".jpg")]
        random.shuffle(frames)  # Ensure randomness

        # Pick exactly 6000 frames
        selected_frames = frames[:max_frames_per_class]

        # Copy only selected frames to final dataset
        for frame in selected_frames:
            src_path = os.path.join(class_folder, frame)
            dest_path = os.path.join(output_class_folder, frame)
            shutil.copy(src_path, dest_path)

print("Final dataset created with EXACTLY 6000 frames per class in 'ucf20_frames'.")



# CODE TO COUNT DATASET SIZE
def count_images(dataset_path):
    total_images = 0
    for dirpath, _, filenames in os.walk(dataset_path):
        total_images += len([file for file in filenames if file.endswith(('.jpg', '.png', '.jpeg'))])

    return total_images


dataset_path = r"C:\Users\sofia\Desktop\Year 3\FYP\Models\.venv\ucf20_frameS"  # Change this to your dataset folder
num_images = count_images(dataset_path)
print(f"Total number of images: {num_images}")


# CODE TO SHUFFLE VIDEOS!!! ------------------------------------------------------

# import os
# import random
#
# # Path to UCF20 dataset containing class subfolders
# ucf20_videos_path = r"C:\Users\sofia\Desktop\Year 3\FYP\Models\.venv\UCF20"  # CHANGE PATH
#
# # Shuffle videos in each class folder
# for class_name in os.listdir(ucf20_videos_path):
#     class_folder = os.path.join(ucf20_videos_path, class_name)
#
#     if os.path.isdir(class_folder):
#         videos = [f for f in os.listdir(class_folder) if f.endswith((".mp4", ".avi"))]
#         random.shuffle(videos)  # Shuffle the list of video filenames
#
#         # **First Pass: Rename to temporary names**
#         for i, video in enumerate(videos):
#             old_path = os.path.join(class_folder, video)
#             temp_path = os.path.join(class_folder, f"temp_{i:06d}{os.path.splitext(video)[1]}")
#             os.rename(old_path, temp_path)
#
#         # **Second Pass: Rename to final names**
#         for i, temp_video in enumerate(sorted(os.listdir(class_folder))):
#             temp_path = os.path.join(class_folder, temp_video)
#             new_path = os.path.join(class_folder, f"video_{i+1:06d}{os.path.splitext(temp_video)[1]}")
#             os.rename(temp_path, new_path)
#
# print("✅ Shuffling complete! Videos in each class folder are now randomized.")








