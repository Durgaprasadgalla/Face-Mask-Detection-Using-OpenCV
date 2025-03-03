import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Define absolute base directory
base_dir = r"C:\Users\Sridh\Desktop\Face-Mask-Detection"
prototxtPath = os.path.join(base_dir, "face_detector", "deploy.prototxt")
weightsPath = os.path.join(base_dir, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
maskModelPath = os.path.join(base_dir, "mask_detector.h5")

# Verify file existence
for path in [prototxtPath, weightsPath, maskModelPath]:
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        exit()
    if os.path.getsize(path) == 0:
        print(f"[ERROR] File is empty: {path}")
        exit()

# Load face detector
print("[INFO] Loading face detector...")
try:
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    if faceNet.empty():
        raise ValueError("Face detection model is empty")
except Exception as e:
    print(f"[ERROR] Failed to load face detector: {e}")
    exit()

# Load mask detector
print("[INFO] Loading mask detector...")
maskNet = load_model(maskModelPath)

def detect_and_predict_mask(image, faceNet, maskNet):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(f"[DEBUG] Total detections: {detections.shape[2]}")

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        print(f"[DEBUG] Detection {i}: Confidence = {confidence:.4f}")
        if confidence > 0.2:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            if startX >= endX or startY >= endY:
                print(f"[DEBUG] Invalid box: {startX, startY, endX, endY}")
                continue

            face = image[startY:endY, startX:endX]
            if face.size == 0:
                print(f"[DEBUG] Empty face region")
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        print(f"[DEBUG] Raw predictions: {preds}")

    return (locs, preds)

# Hardcode the image path (change this to test different images)
image_path = os.path.join(base_dir, "dataset", "with_mask", "69.jpg")
print(f"[INFO] Using image path: {image_path}")

# Load image
image = cv2.imread(image_path)
if image is None:
    print(f"[ERROR] Could not load image from {image_path}")
    print(" - Check if the file exists and the path is correct.")
    print(" - Example paths: dataset/with_mask/2.jpg or dataset/without_mask/augmented_image_240.jpg")
    exit()

print(f"[INFO] Processing image: {image_path}")
(locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)

if len(locs) == 0:
    print("[INFO] No faces detected")
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
        print(f"[INFO] Result: {label_text}")
        cv2.putText(image, label_text, (startX, startY - 10 if startY - 10 > 10 else startY + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        # Define output image path
output_path = os.path.join(base_dir, "output", "result.jpg")

# Create 'output' directory if it doesn't exist
os.makedirs(os.path.join(base_dir, "output"), exist_ok=True)

# Save the output image, replacing the previous one
cv2.imwrite(output_path, image)
print(f"[INFO] Output image saved at: {output_path}")


       
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()