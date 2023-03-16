from flask import Flask, request, Response
import cv2

app = Flask(__name__)

@app.route('/send-data', methods=['POST'])
def send_data():
    data = request.json  # get the JSON data from the request
    print(data)
    # process the data and send it to Node-RED
    # ...
    return 'Data received'

@app.route('/', methods=['GET'])
def home():
    # data = request.json  # get the JSON data from the request
    # print(data)
    # process the data and send it to Node-RED
    # ...
    return 'Home page'

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)