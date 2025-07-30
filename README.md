# Real-Time Object Detection

This project uses a webcam to detect and label objects in real-time using a pre-trained YOLO or TensorFlow model.

## Features
- Real-time object detection from webcam
- Bounding boxes and labels for detected objects

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This command will read the `requirements.txt` file in the current folder and install all the listed packages.
2. Run the main script:
   ```bash
   python main.py
   ```

## Requirements
- Python 3.8+
- OpenCV
- TensorFlow or PyTorch (for model loading)
- Pre-trained model weights (YOLO or TensorFlow) 
C:\Users\swaya\OneDrive\Documents\PythonProjects\real_time_object_detection>python yolov8_detector.py