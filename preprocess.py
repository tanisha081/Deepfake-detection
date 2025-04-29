import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm

# Paths
DATASET_PATH = "data/"
OUTPUT_PATH = "frames/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Initialize Face Detector
detector = MTCNN()


def extract_frames(video_path, output_folder, frames_per_video=10):
    """Extracts frames from a video and detects faces."""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:  # Ensure video is valid
        print(f"Skipping invalid video: {video_path}")
        return

    step = max(1, frame_count // frames_per_video)

    for i in range(frames_per_video):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Skipping frame {i} in {video_path} (invalid frame).")
            continue

        faces = detector.detect_faces(frame)
        if faces:
            x, y, w, h = faces[0]['box']
            if w > 0 and h > 0:  # Ensure valid face bounding box
                face = frame[y:y + h, x:x + w]

                # Check if the face is valid (not empty)
                if face.size == 0:
                    print(f"Skipping empty face from {video_path} at frame {i}")
                    continue

                face = cv2.resize(face, (224, 224))
                frame_path = os.path.join(output_folder, f"{os.path.basename(video_path)}_frame{i}.jpg")
                cv2.imwrite(frame_path, face)

    cap.release()


# Clean up empty image files
for img_file in os.listdir(OUTPUT_PATH):
    img_path = os.path.join(OUTPUT_PATH, img_file)

    # Ensure it's a file before deleting
    if os.path.isfile(img_path) and os.path.getsize(img_path) == 0:
        print(f"Removing empty image: {img_path}")
        os.remove(img_path)

# Process all videos in "real" and "fake" folders
for category in ["real", "fake"]:
    category_path = os.path.join(DATASET_PATH, category)
    output_category_path = os.path.join(OUTPUT_PATH, category)
    os.makedirs(output_category_path, exist_ok=True)

    if not os.path.exists(category_path):
        print(f"Skipping missing directory: {category_path}")
        continue

    for video_file in tqdm(os.listdir(category_path)):
        video_path = os.path.join(category_path, video_file)
        if video_file.endswith((".mp4", ".avi", ".mov")):
            extract_frames(video_path, output_category_path)
        else:
            print(f"Skipping non-video file: {video_file}")
