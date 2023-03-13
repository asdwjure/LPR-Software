from flask import Flask,render_template,Response
import cv2
from queue import Queue


class LPR_Webapp:

    # frame = None
    stream_queue = None

    def __init__(self, stream_queue):
        LPR_Webapp.stream_queue = stream_queue
        
    def start(self):
        app.run(debug=False)

    @classmethod
    def put_frame(cls, frame):
        try:
            cls.stream_queue.put(frame, block=False)
        except Exception as e:
            # print(e)
            pass

    @classmethod
    def generate_frames(cls):
        while True:
            frame = cls.stream_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(LPR_Webapp.generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
