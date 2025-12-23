import cv2  
from flask import Flask, render_template, Response, request, jsonify
import serial
import glob
import time
from flask_socketio import SocketIO

app = Flask(__name__)
# Initialize SocketIO. "cors_allowed_origins" allows connection from any browser.
socketio = SocketIO(app, cors_allowed_origins="*")

arduino = None
try:
    # Find Arduino port
    arduino_ports = glob.glob('/dev/serial/by-id/*Arduino*')
    if arduino_ports:
        arduino_port = arduino_ports[0]
        arduino = serial.Serial(arduino_port, 9600)
        time.sleep(2) # Wait for connection
        print(f"Arduino connected on: {arduino_port}")
    else:
        print("Warning: Arduino not found. Manual control will be disabled.")
except Exception as e:
    print(f"Serial Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    # 0 is usually the default index for the Pi Camera
    camera = cv2.VideoCapture(0)
    
    # Lower resolution for better streaming performance on Pi
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 20)
    if not camera.isOpened():
        raise RuntimeError("Could not start camera.")

    try:
        while True:
            success, frame = camera.read()  # Read the camera frame
            if not success:
                break
            else:
                # Optional: Flip frame if camera is mounted upside down
                # frame = cv2.flip(frame, -1) 

                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                
                # Concat frame one by one and show result
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release() # Release camera resource if loop breaks

@app.route('/video_feed')
def video_feed():
    try:
        return Response(gen_frames(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Camera Error: {e}")
        return "Camera connection failed", 503

@socketio.on('stop_manual_control')
def handle_emergency_stop():
    print("Stop Manual Control!")
    if arduino:
        # 1. Clear any 'Up/Down' commands waiting in the queue
        arduino.reset_output_buffer()
        
        # 2. Send the stop command repeatedly to be safe
        arduino.write(b'X')
        time.sleep(0.05)
        arduino.write(b'X') 
        
        # 3. Optional: Flush ensures data is physically sent
        arduino.flush()

@socketio.on('control')
def handle_control(data):
    if not arduino:
        print("Arduino not connected, ignoring command.")
        return

    command = data.get('command')
    # Map command words to bytes
    controls = {
        'up': b'F',
        'down': b'B',
        'left': b'L',
        'right': b'R',
        'stop': b'X'
    }

    if command in controls:
        try:
            arduino.write(controls[command])
            # Optional: Print to console to verify speed
            print(f"Sent: {command}") 
        except Exception as e:
            print(f"Serial write failed: {e}")

if __name__ == '__main__':
    # CRITICAL CHANGE: Use socketio.run instead of app.run
    socketio.run(app, host='0.0.0.0', debug=True, allow_unsafe_werkzeug=True)