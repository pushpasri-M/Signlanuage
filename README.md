# Project Title: Real-Time Sign Language Recognition System Using Deep Learning and Sensor Fusion
# 🧠 Sign Language Recognition System

A real-time Sign Language Recognition System using deep learning and sensor fusion with MediaPipe, Flex Sensors, and IMU (MPU6050). This project helps bridge the communication gap between hearing-impaired individuals and others by converting hand gestures into spoken words and text.

---

## 📌 Features

- 🔴 Real-time gesture recognition
- 🧩 Sensor fusion: Flex sensors + MPU6050 + MediaPipe hand landmarks
- 📊 Hybrid deep learning model (1D-CNN + BiLSTM)
- 🗣️ Voice output using `pyttsx3`
- 🖥️ Live webcam feed and gesture display
- 🧠 Trained on custom dataset using PyTorch
- 🧪 Accurate recognition of dynamic and static gestures

---

## 🛠️ Technologies Used

| Category              | Tools/Technologies                             |
|-----------------------|------------------------------------------------|
| Programming Language  | Python, Arduino C                              |
| Libraries/Frameworks  | PyTorch, OpenCV, MediaPipe, pyttsx3, NumPy     |
| Microcontrollers      | ESP32 (with Arduino IDE)                       |
| Sensors               | Flex Sensor, MPU6050 IMU                       |
| ML Algorithms         | 1D-CNN, BiLSTM, RandomForest (initial version)|
| Data Handling         | pandas, joblib, scikit-learn                   |

---

## 🔧 Setup Instructions

### 1. Hardware Connections

- Connect **Flex Sensors** to analog pins of ESP32.  
- Connect **MPU6050 (IMU)** via I2C (SCL/SDA) to ESP32.  
- Ensure ESP32 sends serial data in a structured format (e.g., comma-separated).

### 2. Software Installation


# Clone the repo
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition

# Install Python dependencies
pip install -r requirements.txt

# Data Flow Overview
Sensors collect finger bend and wrist movement data.

MediaPipe captures 3D hand landmarks.

Combined data is processed through a hybrid model (1D-CNN + BiLSTM).

Predicted gesture is translated into text and spoken via pyttsx3.



## 🚀 Project Structure

```bash
sign-language-recognition/
│
├── model/
│   ├── gesture_model.pkl         # Trained RandomForest or CNN+BiLSTM model
│   ├── scaler.pkl                # StandardScaler for input normalization
│   ├── label_encoder.pkl         # For decoding predicted classes
│   └── feature_names.txt         # Ordered feature names for prediction
│
├── arduino/
│   └── flex_imu_sender.ino       # Arduino code for sending sensor data
│
├── dataset/
│   └── *.csv                     # Labeled sensor + MediaPipe data
│
├── main.py                       # Main Python script for real-time recognition
├── train_model.py                # Script to train the ML/DL model
├── requirements.txt              # All dependencies
└── README.md                     # Project documentation


