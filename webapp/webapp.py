###################################################################################################
# Title: Web application for license plate recognition software
# Description: Starts the Flask web server on http://localhost:5000/ where the live feed and
# plates entries are displayed. Also uses a SQL database from Falsk to store entered plates.
# Date: 16.3.2023
# 
# Jure Rebernik magistrska naloga
###################################################################################################

from flask import Flask, render_template, Response, flash, request
from flask_sqlalchemy import SQLAlchemy
import cv2

app = Flask(__name__)
app.secret_key = '1234'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///text_entries.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.app_context().push()

db = SQLAlchemy(app)

class TextEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(20))

class Webapp:

    stream_queue = None

    def __init__(self, stream_queue):
        Webapp.stream_queue = stream_queue
        
    @classmethod
    def start(cls):
        db.create_all()
        app.run(host='0.0.0.0', port=5000, debug=False)

    @staticmethod
    @app.route('/', methods=['GET', 'POST'])
    def index():
        log_msg = ''

        if request.method == 'POST':
            text = request.form['text']

            if len(text) > 0 and len(text) < 13:
                new_entry = TextEntry(text=text)
                db.session.add(new_entry)
                db.session.commit()
                log_msg = 'Successfully added new plate.'
            else:
                log_msg = 'Invalid plate!'
        
        entries = TextEntry.query.all()
        return render_template('index.html', entries=entries, log_msg=log_msg)

    @staticmethod
    @app.route('/delete/<int:id>', methods=['POST'])
    def delete_entry(id):
        entry = TextEntry.query.filter_by(id=id).first()
        db.session.delete(entry)
        db.session.commit()
        return '', 204

    @staticmethod
    def gen_frames():
        while True:
            try:
                frame = Webapp.stream_queue.get()
            except Exception:
                pass

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    @staticmethod
    @app.route('/video_feed')
    def video_feed():
        return Response(Webapp.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @classmethod
    def put_frame(cls, frame):
        if Webapp.stream_queue != None:
            if not Webapp.stream_queue.full():
                Webapp.stream_queue.put(frame, block=False)

    @classmethod
    def get_platesInDatabase(cls):
        entries = TextEntry.query.all()
        return [entry.text for entry in entries]
