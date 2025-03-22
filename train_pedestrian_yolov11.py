from ultralytics import YOLO

# Load a model
model = YOLO("weights/yolo11s.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="data/pedestrian_traffic_light_yolov11/data.yaml", epochs=50)

success = model.export(format="onnx")