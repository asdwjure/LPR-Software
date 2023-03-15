from flask import Flask, render_template, Response, flash, request
from flask_sqlalchemy import SQLAlchemy
import cv2


class LPR_Webapp:
    app = Flask(__name__)
    app.secret_key = '1234'

    stream_queue = None
    plate_queue = None
    lpr_object = None

    plate = ''

    def __init__(self, lpr_object, stream_queue):
        LPR_Webapp.stream_queue = stream_queue
        LPR_Webapp.lpr_object = lpr_object
        
    def start(self):
        LPR_Webapp.app.run(debug=False)

    @classmethod
    def put_frame(cls, frame):
        if not cls.stream_queue.full():
            cls.stream_queue.put(frame, block=False)

    @classmethod
    def generate_frames(cls):
        while True:
            try:
                frame, LPR_Webapp.plate = cls.stream_queue.get()
            except Exception:
                pass

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



    @app.route('/', methods=['POST', 'GET'])
    def index():
        return render_template('index.html')

    @app.route('/video')
    def video():
        return Response(LPR_Webapp.generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/get_plate')
    def get_plate():
        return LPR_Webapp.plate
