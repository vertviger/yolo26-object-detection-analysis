from ultralytics import YOLO
import os

if __name__ == '__main__':

    # Configutation variables
    
    model_name = "yolo26n.pt" # model config
    
    video_size = 1280 # video size
    daytime = "night_enhanced"  # video time
    
    data_yaml = f"data.yaml" # yaml path

    # TODO: Separate paths for different model sizes
    trained_model_path = "runs/detect/train/weights/best.pt"

    if not os.path.exists(trained_model_path):
        # model training
        base_model = YOLO(model_name)
        base_model.train(
            data = data_yaml,
            epochs = 15, 
            imgsz = 640,
            batch = 16, 
            device = "cuda",
        )
    video_path = f"videos/tollbooth_{daytime}_{video_size}.mp4"
    model = YOLO(trained_model_path)
    results = model.predict(
        source = video_path,
        save = True, 
        conf = 0.3,
        imgsz = video_size,
        device = "cuda",
        save_txt = False,
        save_conf = True
    )

    metrics = model.val(
        data = data_yaml,
        imgsz = video_size,
        conf = 0.3,
        iou = 0.5
    )

    print(f"Results for {daytime} at {video_size}p:")
    print(f"Mean Average Precision: {metrics.box.map:.4f}")
    print(f"Mean Precision: {metrics.box.mp:.4f}")
    print(f"Mean Recall: {metrics.box.mr:.4f}")