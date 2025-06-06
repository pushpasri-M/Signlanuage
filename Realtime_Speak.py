import serial
import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import torch
import torch.nn as nn
import pyttsx3  # Text-to-speech

# === Load scaler, label encoder, and feature names ===
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
with open("feature_names.txt") as f:
    feature_names = [line.strip() for line in f.readlines()]

# === Define model structure ===
class SignLanguageClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SignLanguageClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# === Load model ===
input_dim = len(feature_names)
num_classes = len(label_encoder.classes_)
model = SignLanguageClassifier(input_dim, num_classes)
model.load_state_dict(torch.load("sign_language_model.pth", map_location=torch.device('cpu')))
model.eval()

# === Setup Serial Connection ===
try:
    ser = serial.Serial('COM6', 115200, timeout=1)
    time.sleep(2)
    print("✅ Serial connection established")
except serial.SerialException as e:
    print(f"❌ Serial connection failed: {e}")
    exit()

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# === Setup Text-to-Speech ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)
last_spoken = ""
last_spoken_time = time.time()

# === Start Webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

def extract_landmarks(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        return [coord for point in lm.landmark for coord in (point.x, point.y, point.z)]
    return [0.0] * (21 * 3)

print("🎥 Starting real-time gesture recognition (PyTorch). Press 'q' to quit.")

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            continue

        landmark_data = extract_landmarks(frame)

        line = ser.readline().decode('utf-8').strip()
        parts = line.split(',')
        if len(parts) != 11:
            print("⚠️ Invalid serial input:", line)
            continue

        arduino_data = list(map(float, parts))
        flex_scaled = [val * 0.1 for val in arduino_data[:5]]
        imu_data = arduino_data[5:]
        full_input = flex_scaled + imu_data + landmark_data

        if len(full_input) != len(feature_names):
            print(f"⚠️ Feature count mismatch: {len(full_input)} vs {len(feature_names)}")
            continue

        input_df = pd.DataFrame([full_input], columns=feature_names)
        X_input = scaler.transform(input_df)
        X_tensor = torch.tensor(X_input, dtype=torch.float32)

        # === Inference ===
        with torch.no_grad():
            logits = model(X_tensor)
            predicted_idx = torch.argmax(logits, dim=1).item()
            predicted_label = label_encoder.inverse_transform([predicted_idx])[0]

        # === Speak once every 5 seconds ===
        current_time = time.time()
        if predicted_label != last_spoken or current_time - last_spoken_time >= 5:
            print(f"🗣️ Speaking: {predicted_label}")
            engine.say(predicted_label)
            engine.runAndWait()
            last_spoken = predicted_label
            last_spoken_time = current_time

        # === Display result ===
        cv2.putText(frame, f'Gesture: {predicted_label}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture Recognition - PyTorch", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exiting...")
            break

    except Exception as e:
        print("❌ Runtime error:", e)

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
ser.close()

