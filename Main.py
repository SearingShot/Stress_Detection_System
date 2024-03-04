from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model("best_model.h5")

# Load face cascade
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion_and_stress(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
        predicted_emotion = emotions[max_index]

        stress_status = ""
        if predicted_emotion in ['angry', 'disgust', 'fear', 'sad']:
            stress_status = "Stressed"
        else:
            stress_status = "Not Stressed"

        cv2.putText(test_img, f"Emotion: {predicted_emotion}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(test_img, f"Stress: {stress_status}", (int(x), int(y + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return test_img

def gen_frames():  # generate frame by frame from camera
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame = cv2.flip(frame, 1)  # Flip horizontally
            frame = detect_emotion_and_stress(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
