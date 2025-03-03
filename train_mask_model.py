import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

data = np.load("face_mask_data.npy")
labels = np.load("face_mask_labels.npy")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

head = base_model.output
head = Flatten()(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(1, activation="sigmoid")(head)

model = Model(inputs=base_model.input, outputs=head)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

model.save("mask_detector.h5")
print("[INFO] Model saved as mask_detector.h5")