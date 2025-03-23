from ultralytics import YOLO
from pathlib import Path

model = YOLO("runs/detect/train9/weights/best.pt")

val_imgs_root = Path("data/merged_yolov11/valid/images")
val_imgs = val_imgs_root.glob("*.jpg")

results = model([str(i) for i in val_imgs])

for i, r in enumerate(results):
    r.save(filename=f"preds/{i}.jpg")