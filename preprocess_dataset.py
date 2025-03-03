import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

dataset_path = "dataset"
with_mask_path = os.path.join(dataset_path, "with_mask")
without_mask_path = os.path.join(dataset_path, "without_mask")

data = []
labels = []

for folder, label in [(with_mask_path, 1), (without_mask_path, 0)]:
    for image_file in os.listdir(folder):
        image_path = os.path.join(folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Could not load {image_path}")
            continue
        
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img_to_array(image)
        image = preprocess_input(image)
        
        data.append(image)
        labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

np.save("face_mask_data.npy", data)
np.save("face_mask_labels.npy", labels)
print(f"[INFO] Processed {len(data)} images")