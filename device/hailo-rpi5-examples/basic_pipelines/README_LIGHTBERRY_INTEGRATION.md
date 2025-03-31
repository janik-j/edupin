# Lightberry Integration with Hailo Detection

This README explains how to integrate the Hailo detection pipeline with the Lightberry voice assistant system to send detection events to a Lightberry server.

## Overview

This integration allows the Hailo detection pipeline to:

1. Detect objects using the Hailo NPU
2. Filter detections based on criteria (size, confidence, etc.)
3. Send detection events to a Lightberry server
4. Authenticate with the Lightberry server using secure device authentication

## Prerequisites

- Raspberry Pi 5 with Hailo NPU
- Lightberry client repository (properly set up)
- Python 3.7+
- Internet connection to reach the Lightberry server

## Directory Structure

The integration assumes the following directory structure:

```
/path/to/
├── hailo-rpi5-examples/
│   └── basic_pipelines/
│       ├── detection.py                  # Modified detection script
│       ├── setup_detection_lightberry.py # Setup script
│       └── ...
└── lightberry-client/
    ├── setup_device.py
    ├── register_device_cli.py
    ├── audio_client.py
    ├── models/
    │   └── Light-berry_en_raspberry-pi_v3_0_0.ppn
    └── ...
```

## Setup Instructions

### 1. Prepare the Environment

Make sure you have both the Hailo examples repository and the Lightberry client repository cloned in the correct locations (as shown in the directory structure above).

### 2. Set Up the Lightberry Device

This step creates the necessary device authentication keys and device ID that will be used to authenticate with the Lightberry server.

```bash
python setup_detection_lightberry.py --setup --device-name "Hailo Detection Device"
```

This will:
- Create a device ID
- Generate RSA key pair for secure authentication
- Save the device name
- Generate registration information needed for the Lightberry server

### 3. Register the Device with the Lightberry Server

Register your device with the Lightberry server using your user credentials:

```bash
python setup_detection_lightberry.py --register --server your-lightberry-server.com --username your-username --password your-password
```

For admin registration:

```bash
python setup_detection_lightberry.py --register --server your-lightberry-server.com --username admin --password admin-password --admin
```

### 4. Run the Detection with Lightberry Integration

Once set up and registered, run the detection script with Lightberry integration:

```bash
python setup_detection_lightberry.py --run
```

You can also run all steps in sequence:

```bash
python setup_detection_lightberry.py --setup --device-name "Hailo Detection Device" --register --server your-lightberry-server.com --username your-username --password your-password --run
```

## How It Works

1. The modified `detection.py` script initializes a WebSocket connection to the Lightberry server
2. It authenticates using the device credentials created during setup
3. When objects are detected (like traffic signs, people, etc.), the script sends detection events to the server
4. The server can then process these events and respond accordingly (e.g., by triggering actions or storing the data)

## Detection Event Format

Events sent to the Lightberry server follow this format:

```json
{
  "type": "event",
  "event_type": "detection",
  "label": "Red traffic light",
  "confidence": 0.95,
  "timestamp": 1679590123.45
}
```

## Customization

### Modifying Detection Triggers

To change which detections trigger events, modify the `app_callback` function in `detection.py`. The current implementation sends events for traffic-related objects like traffic lights, stops signs, and crosswalks.

### Adjusting Detection Thresholds

To change the size threshold for triggering detections, modify the `min_bbox_size` parameter in the `user_app_callback_class` constructor. The default is 0.05 (5% of the frame area).

### Cooldown Time

By default, each object type can only trigger once every 40 seconds. Adjust the `cooldown_time` parameter in the `user_app_callback_class` constructor to change this behavior.

## Troubleshooting

### Authentication Issues

If you encounter authentication issues:
1. Check that the device is properly registered with the Lightberry server
2. Verify that the credentials in `.env` file are correct
3. Try re-registering the device

### Connection Issues

If the script cannot connect to the Lightberry server:
1. Check your internet connection
2. Verify the server address in the `.env` file
3. Make sure the Lightberry server is running and accessible
4. Check any firewalls or network restrictions

### Missing Dependencies

If you encounter missing dependencies, install them with:

```bash
pip install websockets asyncio cryptography dotenv
```

## License

This integration is provided under the same license as the Hailo examples and Lightberry client. 