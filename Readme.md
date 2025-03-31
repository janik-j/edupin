# EduPin 
 ![Hackathon Winner](https://img.shields.io/badge/Winner-European%20Builder%20League%20(Munich)%20%7C%20EF%20Hackathon-gree)  


## Overview

EduPin is an educational platform built on Raspberry Pi hardware designed specifically for children's traffic safety. The system leverages AI capabilities through the Hailo AI Hat for Raspberry Pi 5 to detect and announce traffic signals and road signs in real-time, helping children navigate potentially dangerous traffic situations safely. You can also chat with the EduPin in realtime using the wakeword "EduPin".

## Child Safety Features 🛡️

EduPin acts as a safety companion that:
- Detects and announces traffic signals (red lights, green lights)
- Identifies crosswalks and pedestrian crossings
- Provides timely audio alerts about traffic conditions
- Helps children develop traffic awareness in a safe, guided way

These features make EduPin an excellent educational tool for teaching traffic safety basics while providing an extra layer of protection for children navigating streets and intersections.


## Hardware Components 🛠️

- Raspberry Pi 5
- Hailo AI Hat for accelerated machine learning inference
- Camera module for real-time detection
- Speaker for audio feedback
- Additional sensors and peripherals (as applicable)

## Repository Structure 📁

Everything in the `/device` directory is deployed on the Raspberry Pi:

- `/device/lightberry-client/` - Lightberry client for handling real time voice 
- `/device/hailo-rpi5-examples/` - Examples showcasing Hailo AI Hat capabilities on RPi5, including traffic sign detection

## How It Works

The core of EduPin's safety system is the traffic detection module that:

1. Uses the Raspberry Pi camera to continuously monitor the environment
2. Processes video frames through the Hailo AI accelerator
3. Identifies important traffic elements like lights, crosswalks, and signs
4. Triggers appropriate voice announcements when safety-critical objects are detected
5. Helps children understand when it's safe to cross streets or navigate traffic situations

The system is designed to be:
- Portable - can be carried by a child or attached to mobility devices
- Responsive - provides real-time feedback with minimal latency
- Reliable - uses advanced AI to ensure accurate detection
- Child-friendly - delivers clear, simple audio instructions

## Hailo AI Acceleration 🧠

The Hailo AI Hat provides hardware acceleration for machine learning models, enabling:
- Real-time object detection
- Computer vision applications
- Efficient AI inference at the edge

This acceleration is critical for providing timely safety alerts without delays that could endanger children.

## Datasets
The model is trained on specialized datasets:
1. Traffic Lights Dataset: [Pedestrian Traffic Light Dataset](https://universe.roboflow.com/ono-gedd7/pedestrian-traffic-light-puf4a)
2. Traffic Signs Dataset: [Traffic Signs Dataset](https://universe.roboflow.com/an-wklgs/test-zif7v)

## Model
Deployed in ONNX format enabling:
- **Cross-Framework & Hardware Compatibility**: ONNX allows deployment across multiple frameworks (PyTorch, TensorFlow, etc.) and accelerators (GPUs, TPUs, CPUs), ensuring broad support beyond Hailo's AI processors.
- **Optimized Inference with Quantization & Pruning**: ONNX supports advanced optimization techniques such as structured/unstructured pruning, weight clustering, and post-training quantization (PTQ/QAT), reducing model size and latency while preserving accuracy.
- **Efficient Execution with Graph Optimizations**: ONNX provides operator fusion, constant folding, and redundant node elimination, optimizing computational graphs for faster inference across different runtime environments (e.g., ONNX Runtime, TensorRT).

## Getting Started 🏁

### Prerequisites
- Raspberry Pi 5
- Hailo AI Hat properly connected
- Raspberry Pi OS (64-bit recommended)
- Python 3.7+
- Raspberry Pi Camera module
- Speaker for audio output

### Installation

1. Clone this repository to your Raspberry Pi:
   ```bash
   git clone https://github.com/yourusername/EduPin.git
   cd EduPin
   ```

2. Install neccessary software in /device/ on Raspberry Pi 5

## Usage Examples

Run the traffic safety detection system:

```bash
cd device/hailo-rpi5-examples/basic_pipelines
python detection.py
```

The system will start monitoring the environment and providing safety alerts when traffic signals are detected.
If you want to enable custom wake word detection make sure to include a "your-custom.ppn" wake word model in /lightberry-client/models.

## Documentation 📚

- [Hailo AI Hat Documentation](https://hailo.ai/developer-zone/)
- [Raspberry Pi 5 Documentation](https://www.raspberrypi.com/documentation/)