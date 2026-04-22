import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import tensorflow as tf
import customtkinter as ctk
from PIL import Image

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.abspath(os.path.join(base_path, ".."))
    return os.path.join(base_path, relative_path)

MODEL_PATH = resource_path("model/dog_model.h5")
CLASSES_PATH = resource_path("model/classes.json")
IMG_SIZE = (224, 224)
PREVIEW_SIZE = (350, 350)

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    model = None
    print(f"Model loading error: {e}")

try:
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)
except Exception as e:
    classes = None
    print(f"Classes loading error: {e}")

if isinstance(classes, dict):
    try:
        classes = [classes[str(i)] for i in range(len(classes))]
    except Exception:
        pass




class DogBreedApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Dog Breed Classifier")
        self.geometry("900x650")
        self.minsize(850, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.selected_image_path = None
        self.current_ctk_image = None

        self.create_ui()

    def create_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.left_frame = ctk.CTkFrame(self, corner_radius=15)
        self.left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(2, weight=1)

        self.title_label = ctk.CTkLabel(
            self.left_frame,
            text="Dog Breed Classifier",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.select_button = ctk.CTkButton(
            self.left_frame,
            text="Choose image of the dog",
            command=self.select_image,
            height=40
        )
        self.select_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.image_label = ctk.CTkLabel(
            self.left_frame,
            text="Image is not selected yet",
            width=350,
            height=350
        )
        self.image_label.grid(row=2, column=0, padx=20, pady=20)

        self.file_label = ctk.CTkLabel(
            self.left_frame,
            text="",
            wraplength=320
        )
        self.file_label.grid(row=3, column=0, padx=20, pady=(0, 10))

        self.right_frame = ctk.CTkFrame(self, corner_radius=15)
        self.right_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        self.right_frame.grid_columnconfigure(0, weight=1)

        self.result_title = ctk.CTkLabel(
            self.right_frame,
            text="Result",
            font=ctk.CTkFont(size=26, weight="bold")
        )
        self.result_title.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.predict_button = ctk.CTkButton(
            self.right_frame,
            text="PREDICT",
            command=self.predict_selected_image,
            height=50,
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.predict_button.grid(row=1, column=0, padx=20, pady=15, sticky="ew")

        self.main_result_label = ctk.CTkLabel(
            self.right_frame,
            text="Main Prediction",
            font=ctk.CTkFont(size=22, weight="bold"),
            wraplength=350,
            justify="center"
        )
        self.main_result_label.grid(row=2, column=0, padx=20, pady=(20, 10))

        self.confidence_label = ctk.CTkLabel(
            self.right_frame,
            text="",
            font=ctk.CTkFont(size=16)
        )
        self.confidence_label.grid(row=3, column=0, padx=20, pady=(0, 20))

        self.top3_box = ctk.CTkTextbox(
            self.right_frame,
            width=350,
            height=250,
            font=("Consolas", 16)
        )
        self.top3_box.grid(row=4, column=0, padx=20, pady=(10, 20), sticky="nsew")
        self.top3_box.insert("1.0", "Top 3 predictions will be here...")
        self.top3_box.configure(state="disabled")

        self.disclaimer_label = ctk.CTkLabel(
            self.right_frame,
            text=(
                "Disclaimer:\n"
                "For best prediction accuracy, try running the prediction multiple times "
                "with images of the dog from different angles (front, side, full body)."
            ),
            font=ctk.CTkFont(size=12),
            text_color="gray",
            wraplength=320,
            justify="center"
        )
        self.disclaimer_label.grid(row=5, column=0, padx=20, pady=(10, 15))

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select image of the dog",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.selected_image_path = file_path
        self.file_label.configure(text=os.path.basename(file_path))
        self.show_preview(file_path)

        self.main_result_label.configure(text="Ready for prediction")
        self.confidence_label.configure(text="")
        self.update_top3_text("Click on PREDICT.")

    def show_preview(self, image_path):
        try:
            pil_img = Image.open(image_path)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=PREVIEW_SIZE)

            self.current_ctk_image = ctk_img
            self.image_label.configure(image=self.current_ctk_image, text="")
        except Exception as e:
            self.image_label.configure(text=f"Error image preview:\n{e}", image=None)

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError("Error loading image.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return img

    def predict_selected_image(self):
        if model is None:
            self.main_result_label.configure(text="Model not loaded")
            self.confidence_label.configure(text="Check model/dog_model.h5")
            self.update_top3_text("Model loading failed.")
            return

        if classes is None:
            self.main_result_label.configure(text="Classes not loaded")
            self.confidence_label.configure(text="Check model/classes.json")
            self.update_top3_text("Classes loading failed.")
            return

        if not self.selected_image_path:
            self.main_result_label.configure(text="First select the image")
            self.confidence_label.configure(text="")
            self.update_top3_text("Click on 'Select picture of dog'.")
            return

        try:
            img = self.preprocess_image(self.selected_image_path)
            prediction = model.predict(img, verbose=0)[0]

            top3 = np.argsort(prediction)[-3:][::-1]

            best_index = top3[0]
            best_breed = classes[best_index]

            self.main_result_label.configure(text=f"Main prediction:\n{best_breed}")

            lines = []
            for rank, idx in enumerate(top3, start=1):
                breed = classes[idx]
                conf = prediction[idx] * 100
                lines.append(f"{rank}. {breed} — {conf:.2f}%")

            self.update_top3_text("\n".join(lines))

        except Exception as e:
            self.main_result_label.configure(text="Error in prediction")
            self.confidence_label.configure(text="")
            self.update_top3_text(str(e))

    def update_top3_text(self, text):
        self.top3_box.configure(state="normal")
        self.top3_box.delete("1.0", tk.END)
        self.top3_box.insert("1.0", text)
        self.top3_box.configure(state="disabled")


if __name__ == "__main__":
    app = DogBreedApp()
    app.mainloop()