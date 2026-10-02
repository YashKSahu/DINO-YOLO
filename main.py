from ultralytics.nn.modules.block import DINOv2
from ultralytics.nn.modules.conv import ConvDummy

from ultralytics import YOLO

# Load the model using our custom YAML
model = YOLO("yolov8-dino.yaml")

# Train
model.train(
    data="coco8.yaml", # or your_custom_dataset.yaml
    epochs=100,
    imgsz=640, # resizing to 644 might help align patches!
    batch=16,
    device=0
)
