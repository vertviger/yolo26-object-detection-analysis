import cv2
import os

# Define paths
video_path = "videos/tollbooth_rain_1280.mp4"
image_output_dir = "datasets/rain/images"
label_output_dir = "datasets/rain/labels"

os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(label_output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Extracting {total_frames} frames from {video_path}...")

frame_idx = 0
image_paths = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Format filename to match frame index (e.g., frame_000000.jpg)
    image_name = f"frame_{frame_idx:06d}.jpg"
    image_path = os.path.join(image_output_dir, image_name)
    
    # Save frame image
    cv2.imwrite(image_path, frame)
    
    # Store relative path for train.txt (using forward slashes for YOLO compatibility)
    relative_path = f"./datasets/rain/images/{image_name}"
    image_paths.append(relative_path + "\n")
    
    frame_idx += 1

cap.release()

# Generate the correct train.txt file
train_txt_path = os.path.join(label_output_dir, "train.txt")
with open(train_txt_path, "w") as f:
    f.writelines(image_paths)

print(f"Successfully extracted {frame_idx} frames and generated {train_txt_path}!")