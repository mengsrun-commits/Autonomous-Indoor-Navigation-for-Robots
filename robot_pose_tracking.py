import math
import time
import serial
import matplotlib.pyplot as plt
import glob
import sys  # Added to allow safe exiting

class PoseTracker:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.path_x = [x]
        self.path_y = [y]

    def update(self, v, omega, dt):
        """Updates pose using linear velocity (v) and angular velocity (omega)."""
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt
        
        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        self.path_x.append(self.x)
        self.path_y.append(self.y)

class RealTimeVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Real-Time Robot Trajectory")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)
        self.line, = self.ax.plot([], [], 'b-', label="Path")
        self.robot_marker, = self.ax.plot([], [], 'ro', markersize=8)
        self.heading, = self.ax.plot([], [], 'r-', linewidth=2)
        self.ax.legend()

    def update_plot(self, tracker):
        self.line.set_data(tracker.path_x, tracker.path_y)
        self.robot_marker.set_data([tracker.x], [tracker.y])
        
        arrow_len = 0.3
        hx = tracker.x + arrow_len * math.cos(tracker.theta)
        hy = tracker.y + arrow_len * math.sin(tracker.theta)
        self.heading.set_data([tracker.x, hx], [tracker.y, hy])
        
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def parse_serial_data(ser):
    """
    Reads the most recent line from serial. 
    Expects format: "v,omega" (e.g., "0.45,0.12")
    """
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            parts = line.split(',')
            if len(parts) == 2:
                v = float(parts[0])
                omega = float(parts[1])
                return v, omega
    except Exception as e:
        print(f"Serial Parsing Error: {e}")
    
    return None, None

# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Auto-detect Arduino
    arduino_connection = None
    BAUD_RATE = 9600
    
    try:
        # Check specific by-id path first (Best for Linux/Pi)
        arduino_ports = glob.glob('/dev/serial/by-id/*Arduino*')
        
        # If not found by ID, try standard USB ports (fallback)
        if not arduino_ports:
            arduino_ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')

        if arduino_ports:
            port = arduino_ports[0]
            print(f"Attempting to connect to: {port}")
            arduino_connection = serial.Serial(port, BAUD_RATE, timeout=1)
            arduino_connection.reset_input_buffer()
            time.sleep(2) # Wait for Arduino to reset/stabilize
            print("Connected successfully!")
        else:
            print("Error: No Arduino found.")
            sys.exit(1) # Stop script here

    except Exception as e:
        print(f"Connection Error: {e}")
        sys.exit(1)

    # 2. Initialize Logic
    tracker = PoseTracker()
    viz = RealTimeVisualizer()
    last_time = time.time()
    
    print("Starting Tracking...")
    
    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            
            # 3. Read Data (Using the corrected variable 'arduino_connection')
            v, omega = parse_serial_data(arduino_connection)
            
            if v is not None and omega is not None:
                tracker.update(v, omega, dt)
                viz.update_plot(tracker)
                last_time = current_time
            
            # Prevent plotting from freezing if data stops coming
            plt.pause(0.001) 

    except KeyboardInterrupt:
        print("\nExiting...")
        if arduino_connection:
            arduino_connection.close()
        plt.close()