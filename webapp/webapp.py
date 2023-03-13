from flask import Flask,render_template,Response
import cv2
from queue import Queue


class LPR_Webapp:
    app = Flask(__name__)

    stream_queue = None

    def __init__(self, lpr_object, stream_queue):
        LPR_Webapp.stream_queue = stream_queue
        self.lpr_object = lpr_object
        
    def start(self):
        LPR_Webapp.app.run(debug=False)

    @classmethod
    def put_frame(cls, frame):
        if not cls.stream_queue.full():
            cls.stream_queue.put(frame, block=False)

    @classmethod
    def generate_frames(cls):
        while True:
            # if not cls.stream_queue.empty():
            try:
                frame = cls.stream_queue.get()
                # frame = self.lpr_object.get_frame()
            except Exception:
                pass

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/video')
    def video():
        return Response(LPR_Webapp.generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
