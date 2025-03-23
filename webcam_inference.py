from ultralytics import YOLO
import cv2
# Load a pretrained YOLO11n model
model = YOLO("best.pt")

# Zoom factor (higher values = more zoom)
zoom_factor = 2.0

# Run inference on the source
cap = cv2.VideoCapture(1)
while True:
    success, frame = cap.read()
    if not success:
        break
        
    # Apply zoom by cropping the center portion
    h, w = frame.shape[:2]
    # Calculate new dimensions
    new_h, new_w = int(h/zoom_factor), int(w/zoom_factor)
    # Calculate starting point for cropping (center crop)
    start_x, start_y = int((w - new_w)/2), int((h - new_h)/2)
    # Crop the frame
    cropped_frame = frame[start_y:start_y+new_h, start_x:start_x+new_w]
    # Resize back to original dimensions
    zoomed_frame = cv2.resize(cropped_frame, (w, h))
    
    # Run inference on the zoomed frame
    results = model(zoomed_frame, stream=False, iou=0.5)
    
    for result in results:
        # Get the annotated frame
        annotated_frame = result.plot()
        
        # Display the annotated frame
        cv2.imshow("custom YOLOv11s", annotated_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()