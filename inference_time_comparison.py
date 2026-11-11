from ultralytics import YOLO

# Configutation variables

# model config
model_name = "yolo26n.pt"
# model_name = "yolo26m.pt"

# video config
video_size = 640
# video_size = 1280
video_path = f"videos/tollbooth_{video_size}.mp4"

model = YOLO(model_name)
results = model.predict(
    source = video_path,
    save = True, 
    conf = 0.3,
    imgsz = video_size,
    device = "cuda",
    save_txt = False,
    save_conf = True
)