import serial
import matplotlib.pyplot as plt

# ====== CHANGE THIS ======
SERIAL_PORT = 'COM3'   # or '/dev/ttyUSB0' on Linux/Mac
BAUD_RATE   = 115200

# ====== INITIALIZE SERIAL ======
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

x_list = []
y_list = []

plt.ion()  # Interactive mode on
fig, ax = plt.subplots()
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Robot Trajectory")
ax.axis('equal')

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if not line:
            continue
        
        # Parse x, y, theta
        try:
            xs, ys, _ = map(float, line.split(','))
        except ValueError:
            continue
        
        x_list.append(xs)
        y_list.append(ys)

        # Clear and plot
        ax.cla()
        ax.plot(x_list, y_list, '-b')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Robot Trajectory")
        ax.axis('equal')
        plt.pause(0.01)

except KeyboardInterrupt:
    print("Plotting stopped.")
    ser.close()
    plt.show()
