// Motor pins
int IN1 = 5;   // Left motor
int IN2 = 6;
int IN3 = 9;   // Right motor
int IN4 = 10;

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  Serial.begin(9600);  // Must match Pi baud rate
  Serial.println("Arduino ready for commands");
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();  // Read one byte from Pi

    switch (cmd) {
      case 'F': forward(); break;   // Forward
      case 'B': backward(); break;  // Backward
      case 'L': left(); break;      // Turn Left
      case 'R': right(); break;     // Turn Right
      case 'X': stopMotor(); break; // Stop
    }
  }
}

// ================= Motor Functions =================

void forward() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

void backward() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
}

void left() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

void right() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
}

void stopMotor() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}