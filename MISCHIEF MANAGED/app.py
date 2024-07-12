from flask import Flask, render_template, Response
import cv2
import numpy as np
import time

app = Flask(__name__)

def gen_frames():
    cap = cv2.VideoCapture(0)
    time.sleep(3)
    for i in range(60):
        ret, background = cap.read()
    background = np.flip(background, axis=1)

    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = np.flip(img, axis=1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 120, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask1 = mask1 + mask2

        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        mask2 = cv2.bitwise_not(mask1)

        res1 = cv2.bitwise_and(img, img, mask=mask2)
        res2 = cv2.bitwise_and(background, background, mask=mask1)

        finalOutput = cv2.addWeighted(res1, 1, res2, 1, 0)
        ret, buffer = cv2.imencode('.jpg', finalOutput)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
