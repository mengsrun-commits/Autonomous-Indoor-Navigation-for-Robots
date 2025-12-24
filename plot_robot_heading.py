import serial
import matplotlib.pyplot as plt
import numpy as np
import time
import glob

# ===== Serial setup =====
BAUD_RATE = 115200   # <-- MUST match Serial.begin() in Arduino

ports = glob.glob("/dev/serial/by-id/*Arduino*")
if not ports:
    raise RuntimeError("Arduino not found")

arduino_port = ports[0]
print(f"Connecting to {arduino_port} ...")

ser = serial.Serial(arduino_port, BAUD_RATE, timeout=1)
time.sleep(2)
print("Connected.\nStarting plot...")

# ===== Plot setup =====
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect("equal")
ax.set_title("Robot Heading")

line, = ax.plot([0, 0], [0, 0])
plt.show(block=False)
plt.pause(0.1)

theta = 0.0
t_prev = time.time()

try:
    while True:
        line_raw = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line_raw:
            continue

        try:
            gz = float(line_raw)   # rad/s from Arduino
        except ValueError:
            continue

        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now

        theta = (theta + gz * dt) % (2*np.pi)

        dx = 0.5 * np.cos(theta)
        dy = 0.5 * np.sin(theta)

        line.set_data([0, dx], [0, dy])
        plt.pause(0.01)

except KeyboardInterrupt:
    print("\nStopping...")
    ser.close()
    plt.show()
