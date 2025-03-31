import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example
        # Dictionary to track detected objects
        self.triggered_objects = {}
        # Minimum size threshold for bounding boxes (as percentage of frame size)
        self.min_bbox_size = 0.05  # 5% of frame size
        # Cooldown time in seconds before an object can be detected again
        self.cooldown_time = 40

    def new_function(self):  # New function example
        return "The meaning of life is: "
    
    def should_trigger(self, label, bbox, width, height):
        """
        Determines if a detection should trigger a notification
        
        Args:
            label: The detection label
            bbox: The bounding box (x, y, w, h)
            width: Frame width
            height: Frame height
            
        Returns:
            bool: True if the detection should trigger a notification
        """
        # Calculate relative bbox size (area)
        bbox_width = bbox.width()
        bbox_height = bbox.height()
        frame_area = width * height
        bbox_area = bbox_width * bbox_height
        bbox_ratio = bbox_area / frame_area
        
        # Check if the bbox is large enough
        if bbox_ratio < self.min_bbox_size:
            return False
            
        # Check if this object has been triggered recently
        current_time = time.time()
        if label in self.triggered_objects:
            last_trigger_time = self.triggered_objects[label]
            if current_time - last_trigger_time < self.cooldown_time:
                return False
                
        # Update the trigger time for this label
        self.triggered_objects[label] = current_time
        return True

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        if label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")
            detection_count += 1
        
        # Check traffic-related objects with a single function
        traffic_objects = {
            "Stop": "Stop",
            "crosswalk": "crosswalk",
            "finger_point": "finger_point",
            "green light": "green light",
            "pedestrian Traffic Light": "pedestrian Traffic Light",
            "red light": "red light",
            "orange light": "orange light"
        }
        
        if label in traffic_objects and user_data.should_trigger(label, bbox, width, height):
            print(f"{traffic_objects[label]} {confidence:.2f}")
            
    if user_data.use_frame:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Example of how to use the new_variable and new_function from the user_data
        # Let's print the new_variable and the result of the new_function to the frame
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    #print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
