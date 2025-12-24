import serial
import matplotlib.pyplot as plt
import numpy as np
import time

# ===== Serial setup =====
SERIAL_PORT = 'COM6'  # Change to your port
BAUD_RATE = 115200

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # wait for Arduino

# ===== Plot setup =====
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.set_title("Robot Heading")

current_arrow = None
theta = 0.0
dt = 0.1  # Arduino integration interval in seconds

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if not line:
            continue
        try:
            gz_rad = float(line)  # read gyroZ in rad/s
        except:
            continue

        # integrate to get heading
        theta += gz_rad * dt
        theta = theta % (2 * np.pi)

        # remove previous arrow
        if current_arrow:
            current_arrow.remove()

        # compute arrow components
        dx = 0.5 * np.cos(theta)
        dy = 0.5 * np.sin(theta)
        current_arrow = ax.arrow(0, 0, dx, dy, head_width=0.05, head_length=0.1, fc='r', ec='r')

        plt.pause(0.01)

except KeyboardInterrupt:
    print("Stopping...")
    ser.close()
    plt.show()
