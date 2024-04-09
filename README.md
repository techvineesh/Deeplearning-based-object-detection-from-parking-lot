# Multi-Camera Vehicle Detection for Parking Management

This project aims to develop a multi-camera system for vehicle detection and parking management using a combination of RCNN (Region-based Convolutional Neural Network), YOLO (You Only Look Once), and AlexNet models. The system will be capable of efficiently detecting vehicles in real-time across multiple camera feeds and providing valuable data for parking lot management.

## Overview

The project combines the strengths of different deep learning models to achieve accurate and fast vehicle detection in various parking scenarios. The use of multiple camera inputs allows for comprehensive coverage of parking areas, enabling efficient management of parking spaces and resources.

## Models Used

1. **RCNN (Region-based Convolutional Neural Network):**
   - RCNN is utilized for its effectiveness in object localization. It segments the input image into regions and proposes bounding boxes for objects, which is crucial for accurate vehicle detection.

2. **YOLO (You Only Look Once):**
   - YOLO is chosen for its real-time detection capabilities. It processes images in a single pass, making it well-suited for rapid vehicle detection across multiple camera streams.

3. **AlexNet:**
   - AlexNet is used for vehicle classification. Once vehicles are detected, AlexNet helps classify the type of vehicle detected (car, truck, motorcycle, etc.), providing additional insights for parking management.

## Project Structure

- `data/`: Contains datasets and data preprocessing scripts.
- `models/`: Stores trained RCNN, YOLO, and AlexNet models.
- `src/`: Source code for the multi-camera detection system.
  - `camera.py`: Camera interface for capturing and processing live video feeds.
  - `detection.py`: Integrates RCNN and YOLO models for vehicle detection.
  - `classification.py`: Uses AlexNet for vehicle classification.
  - `utils/`: Utility functions for image processing, model loading, etc.
- `results/`: Output directory for detected vehicle images and logs.

## Requirements

- Python 3.x
- TensorFlow (for RCNN and AlexNet)
- PyTorch (for YOLO)
- OpenCV
- NumPy

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/multi-camera-vehicle-detection.git
   cd multi-camera-vehicle-detection
   ```

2. **Setup Environment:**
   - Install required packages using `pip`:
     ```bash
     pip install -r requirements.txt
     ```

3. **Download Pre-trained Models:**
   - Download pre-trained weights for RCNN, YOLO, and AlexNet models and place them in the `models/` directory.

4. **Run the System:**
   - Execute the main script to start the multi-camera detection system:
     ```bash
     python src/main.py
     ```

5. **View Results:**
   - Detected vehicles and logs will be saved in the `results/` directory.

## Future Enhancements

- Integration with a centralized parking management system.
- Optimization for edge devices and real-world deployment.
- Implementation of additional features like vehicle counting, speed estimation, etc.

## Contributing

Contributions are welcome! If you have ideas for improvements, please create an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
