#include <Wire.h>
#include <MPU6050_light.h>

MPU6050 imu(Wire);

void setup() {
  Serial.begin(115200);
  Wire.begin();

  // initialize IMU
  imu.begin();
  imu.calcGyroOffsets();  // <-- no arguments in MPU6050_light

  Serial.println("IMU connected successfully!");
}

void loop() {
  imu.update();              // update IMU readings
  float gz_deg = imu.getGyroZ();         // gyro Z in deg/s
  float gz_rad = gz_deg * (3.14159265359 / 180.0); // convert to rad/s

  Serial.println(gz_rad, 6);  // print in rad/s
  delay(100);
}
