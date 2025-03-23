from ultralytics import YOLO

# Load a model
model = YOLO("weights/yolo11s.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="data/merged_yolov11/data.yaml", epochs=100)

success = model.export(format="onnx")