#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

// MPU6050 object
Adafruit_MPU6050 mpu;

// Flex sensor analog pins (adjust as per your board)
const int flexPins[5] = {32, 33, 34, 35, 36};
int flexValues[5];

void setup() {
  Serial.begin(115200); // High baud rate for fast serial transfer
  while (!Serial) delay(10); // Wait for Serial (for native USB boards)

  // Initialize MPU6050
  if (!mpu.begin()) {
    Serial.println("MPU6050 not found!");
    while (1) delay(10);
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  delay(100);
}

void loop() {
  // Read Flex Sensor Data
  for (int i = 0; i < 5; i++) {
    flexValues[i] = analogRead(flexPins[i]);
  }

  // Get IMU readings
  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);

  // Output: F1, F2, F3, F4, F5, Ax, Ay, Az, Gx, Gy, Gz
  for (int i = 0; i < 5; i++) {
    Serial.print(flexValues[i]);
    Serial.print(",");
  }

  Serial.print(accel.acceleration.x, 3); Serial.print(",");
  Serial.print(accel.acceleration.y, 3); Serial.print(",");
  Serial.print(accel.acceleration.z, 3); Serial.print(",");
  Serial.print(gyro.gyro.x, 3); Serial.print(",");
  Serial.print(gyro.gyro.y, 3); Serial.print(",");
  Serial.println(gyro.gyro.z, 3);

  delay(100); // Adjust if needed
}

