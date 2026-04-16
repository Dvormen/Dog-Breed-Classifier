# Dog Breed Classifier

Author: Lukáš Dvořák

E-mail: luky.dvorak07@gmail.com

Date: 16.4.2026

School: Střední průmyslová škola elektrotechnická, Praha 2, Ječná 30

Project: School project – Machine Learning Application

------------------------------------------------------------

PROJECT DESCRIPTION

Dog Breed Classifier is a Python application that uses a neural network to classify dog breeds from images.

The user can:
- Place images into the input/ folder
- Run the application
- See top 3 predicted dog breeds with probabilities

------------------------------------------------------------

TECHNOLOGIES USED

- Python 3.10+
- TensorFlow / Keras
- OpenCV
- NumPy
- MobileNetV2 (transfer learning)

------------------------------------------------------------


INSTALLATION

Install Python 3.10+

Install dependencies:

pip install tensorflow opencv-python numpy

OR (recommended):
pip install -r requirements.txt

------------------------------------------------------------

HOW TO RUN

python app.py

Steps:
1. Put image(s) into input/ folder
2. Press ENTER
3. See prediction results in console

------------------------------------------------------------

DATASET CREATION

Dataset was created manually:

- Videos of dog breeds were collected from publicly available sources (YouTube)
- Frames were extracted using a custom Python script
- Additional images were manually collected using Google Images search
- All images were manually checked and cleaned

IMPORTANT:
No pre-existing datasets (Kaggle, ImageNet, etc.) were used.

Dataset contains:
- 30 dog breeds
- ~3000 images

------------------------------------------------------------

DATA PREPROCESSING

- Resize images to 224x224
- Normalize pixel values to range 0–1
- Remove invalid or low-quality images manually
- Labels created from folder names

------------------------------------------------------------

MODEL

- Base model: MobileNetV2 (transfer learning)
- Custom classification head added
- Trained for 5 epochs
- Training accuracy: ~90%
- Validation accuracy: ~69%

------------------------------------------------------------

USAGE EXAMPLE

1. Add image to input/ folder
2. Run application (python app.py)
3. System outputs top 3 predictions

------------------------------------------------------------

REAL-WORLD USE

- Dog shelter assistance
- Breed identification
- Educational purposes

------------------------------------------------------------

SUMMARY

This project includes:
- Custom dataset creation
- Data preprocessing
- Neural network training
- Deployment application (CLI tool)
