from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    # Placeholder for camera logic
    # If camera fails: raise Exception("Camera not found")
    while True:
        # Your frame capture logic here
        pass

@app.route('/video_feed')
def video_feed():
    try:
        return Response(gen_frames(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return "Camera connection failed", 503

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') # Host 0.0.0.0 allows access from other devices