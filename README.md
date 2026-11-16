# Yolo26-object-detection analysis
<img width="1280" height="720" alt="tollbooth_day_dp_1280_m" src="https://github.com/user-attachments/assets/81afd01d-410a-4114-8f9a-23138d3dc1f0" />

## Specification
* **Task**: Highway cars detection
* **Models Used**: [Yolo](https://docs.ultralytics.com/models/yolo26) 26n and 26m
* **Dataset**: Custom videos filmed in the car simulator BeamNG.drive
* **Model parameters**: Confidence - 0.5, Iou - 0.8, Imgsz - set to video resolution
* **Classes to classify**: Car, Truck, Bus

## Description
The goal of this project is to compare Yolo26 nano and medium models. 
Main measurements are inference time, precision and recall.
The test scenario conducted is traffic scanning, the model has to detect commercial vehicles passing onto the highway.
Testing scenarios include: daylight and night time footage, daylight footage with a different perspective from training footage, and rainy weather footage. 
For full tests report, please check out **yolo26-object-detection-analysis.pdf**

### Inference and Performance Differences:
The YOLOv8n (Nano) variant shows far greater inference throughput, running at ~3.0 ms per image (~333 FPS at 1280p) and scaling up to 1,250 FPS at 640p, 
compared to the Medium model's heavier ~16.3 ms footprint (~60 FPS). 
Furthermore, Nano consistently achieved higher structural recall and mAP50 on standard distributions, 
proving that high speed does not compromise baseline accuracy.

### Best Suited Scenarios:
* **YOLOv8n (Nano)**: Ideal for real-time edge deployment, high-throughput traffic monitoring, and resource-constrained hardware where fast latency and structural generalization are critical priorities.   
* **YOLOv8m (Medium)**: Best suited for complex, data-dense multi-class tracking tasks working in controlled, stationary environments, or scenarios where the training perfectly matches production use and extensive data augmentation can be fully leveraged during training process.

### Why medium model didn't show better results
With ~20 million parameters, the Medium model possesses high capacity, making it prone to overfitting on synthetic rendering artifacts, textures, and geometric priors unique to the training environment. 
Because it memorizes specific details rather than learning generalized coarse shapes, it breaks down under out-of-distribution shifts. 
To perform robustly, it demands heavy data augmentation (domain randomization, lighting variations, and weather blending) as well as longer training process with fine-tuning fitting the testing environment.

## Run Instructions
    clone the repository via **git clone**
    install YoLo models **via pip install -U ultralytics**
    run frames_exporter.py for videos you would like to be tested  
    setup and run inference_time_comparison.py with variables set accordingly to images present in datasets/example/images
    run results will be placed in runs/detect/predict/ and runs/detect/val/

### Structure
    datasets/             # Dataset configuration text files and validation images
    videos/               # Test and training footage
    weights_yolo26n/      # Trained Nano weights
    weights_yolo26m/      # Trained Medium weights
    data_*.yaml           # Dataset configuration YAMLs
    frames_exporter.py    # Helper to extract frames for datasets
    inference_time_comparison.py # Evaluation script
    yolo26-object-detection-analysis.pdf # Project report
