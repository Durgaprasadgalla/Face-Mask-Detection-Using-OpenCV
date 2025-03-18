# Face-Mask-Detection-Using-OpenCV
---------------------------------------------------------------------
CSM D2 TEAM-08

**TEAM DETAILS**
--------------------------------------------------------------------
> GALLA DURGA PRASAD (TL)

> TIRUMALASETTI SAI TEJA

> CHINTHALAPUDI MAHESH

> VAKADA SATYANARAYANA SAI

> PANCHADI RAMAKRISHNA

-------------------------------------------------------------------
Introduction

Face Mask Detection is a computer vision application that identifies whether individuals are wearing face masks in real-time. 
It has become crucial in ensuring public health and safety, especially during pandemics. 
This project employs  Deep Learning and OpenCV to develop an accurate and efficient face mask detection system.
Using a pre-trained deep learning model, the system can classify faces as "Mask" or "No Mask" with high accuracy.


Abstract

This project focuses on Face Mask Detection using Convolutional Neural Networks (CNNs) and OpenCV. 
The system captures real-time video feeds, processes the input using a trained model, and classifies faces into two categories: 
With Mask and Without Mask. 
This project has significant applications in enforcing mask-wearing policies in public areas and improving safety measures.


Technology

- Python: Core programming language used for development.
- OpenCV: Handles image and video processing tasks.
- TensorFlow/Keras: Provides deep learning models for face mask classification.
- CNN Model: A pre-trained or custom-built convolutional neural network is used for classification.


Uses and Applications

Face Mask Detection has a wide range of real-world applications, including:
- Public Safety: Ensures compliance with mask-wearing regulations in public places.
- Healthcare Monitoring: Helps in hospitals and clinics to enforce mask mandates.
- Smart Surveillance: Enhances security camera systems with real-time mask detection alerts.
- Workplace Compliance: Ensures employees adhere to mask guidelines in offices and industries.


Steps to Build

1. Data Collection : Use publicly available datasets of masked and unmasked faces.
2. Model Training : Train a CNN model using TensorFlow/Keras with labeled face mask data.
3. Preprocessing : Resize, normalize, and augment images to improve model accuracy.
4. Face Detection : Use OpenCV’s pre-trained Haar cascade or DNN face detector to locate faces.
5. Mask Classification : Apply the trained model to classify detected faces as "Mask" or "No Mask."
6. Real-time Detection : Integrate with a live camera feed to detect masks in real-time.


Work Flow

1. Input: The system captures video frames from a webcam or CCTV feed.
2. Processing: Face detection is performed using OpenCV, followed by classification using a CNN model.
3. Output: The system overlays "Mask" or "No Mask" labels on detected faces and provides alerts if needed.


Conclusion

This project presents an effective Face Mask Detection system that utilizes deep learning and computer vision techniques.By leveraging OpenCV and CNN models, the system can efficiently classify masked and unmasked faces in real-time. 
This technology can be widely implemented for public safety, healthcare, and surveillance, ensuring better enforcement of mask-wearing policies.



Dataset Link : [click the here Go to drive and download the dataset](https://drive.google.com/drive/folders/19Qc5tIlNbOaCjaxOkBYoWuccxsiLlynx?usp=drive_link)


Face Detector Weights: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

Face-Mask-Detection/

├── dataset/  # Your dataset folder

│   ├── with_mask/          # Images of faces with masks

│   └── without_mask/        # Images of faces without masks

├── face_detector/           # Face detection model files     

│   ├── deploy.prototxt         # Face detector architecture

│   └── res10_300x300_ssd_iter_140000.caffemodel       # Face detector weights

├── preprocess_dataset.py     # Script to preprocess dataset
  
├── train_mask_model.py         # Script to train mask detection model

├── detect_mask_image.py        # Script for image-based detection

├── detect_mask_video.py   # Script for real-time video detection

├── face_mask_data.npy          # Preprocessed image data (generated)

├── face_mask_labels.npy       # Preprocessed labels (generated)

└── mask_detector.h5             # Trained mask detection model (generated)



------------
pip install opencv-python numpy imutils tensorflow scikit-learn matplotlib
--------------
**How to Run**
-----------
Download the Dataset: Place the with_mask and without_mask folders in the dataset/ directory.
Preprocess the Data: Run the preprocessing script:
-------
Terminal

python preprocess_dataset.py _____
#Train the Model: Train the CNN model:
-----
Terminal

python train_mask_model.py_____
#Create The Model: Generate the model.h5
-------
Terminal

python detect_mask_image.py_____
#Test on Images: Detect masks in Image:
------
Terminal

python detect_mask_video.py_____
#Test on Video: Detect masks in real-time video:
--------
