# model_utils.py
import numpy as np
from PIL import Image, ImageOps
import joblib

def load_model(path="mnist_logistic_model.pkl"):
    return joblib.load(path)

def preprocess_image(image):
    image = image.convert('L')                     # Convert to grayscale
    image = image.resize((28, 28))                 # Resize to 28x28

    # Convert to numpy and check background color
    img_array = np.array(image)

    # Auto-invert if the background is white
    if img_array.mean() > 127:                     # Mostly white
        img_array = ImageOps.invert(image)
        img_array = np.array(img_array)

    img_array = img_array / 255.0                  # Normalize to [0, 1]
    return img_array.flatten().reshape(1, -1), Image.fromarray((img_array * 255).astype(np.uint8))
