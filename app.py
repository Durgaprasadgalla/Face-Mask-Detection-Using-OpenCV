from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time

app = Flask(__name__)

# Load face detector and mask detector
prototxt_path = "face_detector/deploy.prototxt"
weights_path = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
try:
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)
    mask_net = load_model("mask_detector.h5")
    print("[INFO] Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Function to detect and predict masks (adapted from detect_mask_video.py)
def detect_and_predict_mask(frame, face_net, mask_net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
    preds = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Lowered threshold to match detect_mask_video.py
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            if startX >= endX or startY >= endY:
                continue

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)  # Use MobileNetV2 preprocessing

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32, verbose=0)

    return (locs, preds)

# Video feed generator
def gen_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read frame.")
            break

        # Resize frame for faster processing (match detect_mask_video.py)
        frame = cv2.resize(frame, (400, 300))
        (locs, preds) = detect_and_predict_mask(frame, face_net, mask_net)

        if len(locs) == 0:
            cv2.putText(frame, "No Face Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                mask_prob = pred[0]
                label = "Mask" if mask_prob > 0.5 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                confidence = mask_prob if label == "Mask" else 1 - mask_prob

                label_text = f"{label}: {confidence:.2f}"
                y = startY - 10 if startY - 10 > 10 else startY + 20
                cv2.putText(frame, label_text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Optimize JPEG encoding for faster streaming
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            print("Error: Could not encode frame.")
            break

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    print("[INFO] Starting video feed")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    if file:
        file_path = os.path.join("static", "uploaded_image.jpg")
        file.save(file_path)
        image = cv2.imread(file_path)
        if image is None:
            print("Error: Could not load uploaded image.")
            return render_template('index.html', error="Invalid image file.")

        # Process the image (same as video stream)
        image = cv2.resize(image, (400, 300))
        (locs, preds) = detect_and_predict_mask(image, face_net, mask_net)

        if len(locs) == 0:
            cv2.putText(image, "No Face Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                mask_prob = pred[0]
                label = "Mask" if mask_prob > 0.5 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                confidence = mask_prob if label == "Mask" else 1 - mask_prob

                label_text = f"{label}: {confidence:.2f}"
                y = startY - 10 if startY - 10 > 10 else startY + 20
                cv2.putText(image, label_text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        cv2.imwrite(file_path, image)
        print("[INFO] Image processed and saved")
        timestamp = int(time.time())
        return render_template('index.html', uploaded_image="uploaded_image.jpg", timestamp=timestamp)
    return render_template('index.html', error="No file uploaded.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)