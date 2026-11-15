from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Configutation variables
    model_variant = "n" #model size
    model_base = f"yolo26{model_variant}.pt" # model config
    
    video_size = 1280 # video size
    daytime = "rain"  # video time
    video_path = f"videos/tollbooth_{daytime}_{video_size}.mp4"
    data_yaml = f"data_{daytime}.yaml" # yaml path

    # TODO: Separate paths for different model sizes
    experiment_name = f"train_yolo26{model_variant}"
    trained_model_path = f"runs/detect/{experiment_name}/weights/best.pt"

    if not os.path.exists(trained_model_path):
        # model training
        base_model = YOLO(model_base)
        base_model.train(
            data = data_yaml,
            epochs = 15, 
            imgsz = 1280,
            batch = 4, 
            amp = True,
            device = "cuda",
            name = experiment_name 
        )
    model = YOLO(trained_model_path)
    results = model.predict(
        source = video_path,
        save = True, 
        conf = 0.5,
        imgsz = video_size,
        device = "cuda",
        save_txt = False,
        save_conf = True
    )

    metrics = model.val(
        data = data_yaml,
        imgsz = video_size,
        conf = 0.5,
        iou = 0.8
    )

    print(f"Results for {daytime} at {video_size}p:")
    print(f"Mean Average Precision: {metrics.box.map:.4f}")
    print(f"Mean Precision: {metrics.box.mp:.4f}")
    print(f"Mean Recall: {metrics.box.mr:.4f}")