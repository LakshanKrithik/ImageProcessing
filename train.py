from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Use 'n' for fastest, 's' or 'm' for better accuracy

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
)
