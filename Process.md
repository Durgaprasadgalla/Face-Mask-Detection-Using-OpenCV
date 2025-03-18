Dataset Link : https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset


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
