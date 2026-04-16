import tensorflow as tf
import cv2
import numpy as np
import json
import os
import logging

# Suppress TensorFlow logs
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load trained model
model = tf.keras.models.load_model("model/dog_model.h5")

# Load class labels
with open("model/classes.json", "r") as f:
    classes = json.load(f)

IMG_SIZE = (224, 224)
INPUT_FOLDER = "input"


def predict_image(image_path):
    """
    Predict dog breed for a single image.
    """
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Cannot load image: {image_path}")
        return

    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]

    top3 = np.argsort(prediction)[-3:][::-1]

    print("\n======================")
    print(f"Image: {image_path}")
    print("----------------------")

    for i in top3:
        print(f"{classes[i]}: {prediction[i] * 100:.2f}%")

    print("======================\n")


def process_folder():
    """
    Process all images in input folder.
    """
    files = os.listdir(INPUT_FOLDER)

    images = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not images:
        print("Input folder is empty.")
        return

    for img in images:
        path = os.path.join(INPUT_FOLDER, img)
        predict_image(path)


print("Dog Breed Classifier started")
print("Place images into 'input/' folder")
print("Press ENTER to run prediction")
print("CTRL+C to exit")

while True:
    input("\nPress ENTER to run prediction...")
    process_folder()