import cv2
import mediapipe as mp
import time
import serial
import csv

# === Initialize Serial Connection ===
ser = serial.Serial('COM6', 115200, timeout=1)  # ✅ Adjust COM port as needed

# === Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks
cap = cv2.VideoCapture(0)

# === Prepare CSV Header ===
header = (
    ["timestamp", "label"] +
    [f"F{i+1}" for i in range(5)] +            # Flex sensor data
    ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"] +     # IMU data
    [f"L{j}_{axis}" for j in range(21) for axis in ['x', 'y', 'z']]  # Vision data
)

# === Define the label to log ===
label = "You"  # This is the label you want to log, change it to the desired label
csv_filename = f"{label}.csv"  # Output CSV file based on label

# === Open CSV File for Writing ===
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header row

    # === Data Capture Loop ===
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image")
            break

        timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # === Serial Read ===
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            try:
                parts = list(map(float, line.split(',')))
                if len(parts) == 11:
                    flex_imu_data = parts
                else:
                    continue
            except ValueError:
                continue
        else:
            continue

        # === Extract Vision Data ===
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Only one hand
            vision_data = []
            for lm in hand_landmarks.landmark:
                vision_data.extend([lm.x, lm.y, lm.z])

            # === Draw Hand Landmarks ===
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # === Write to CSV with Label ===
            writer.writerow([timestamp, label] + flex_imu_data + vision_data)
            print(f"✔️ Logged: {label} @ {timestamp}")

        # === Show Video Feed (with Hand Diagram) ===
        cv2.imshow("Hand Tracking", frame)

        # === Handle Keypresses ===
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q' to quit
            print("Exiting...")
            break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
