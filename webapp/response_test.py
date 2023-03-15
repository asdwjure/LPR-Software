from flask import Flask, render_template, request, Response
from flask_sqlalchemy import SQLAlchemy
import cv2

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///text_entries.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.app_context().push()
db = SQLAlchemy(app)

class TextEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(20))

@app.route('/database', methods=['GET', 'POST'])
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
    return render_template('database.html', entries=entries, log_msg=log_msg)

@app.route('/delete/<int:id>', methods=['POST'])
def delete_entry(id):
    entry = TextEntry.query.filter_by(id=id).first()
    db.session.delete(entry)
    db.session.commit()
    return '', 204

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
    db.create_all()
    app.run(debug=True)
