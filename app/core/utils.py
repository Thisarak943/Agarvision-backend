from PIL import Image
import numpy as np

def preprocess_image_pil(image: Image.Image, img_size=(224, 224)) -> np.ndarray:
    """
    Converts image to RGB, resizes, normalizes, and returns shape (1, H, W, 3)
    """
    image = image.convert("RGB")
    image = image.resize(img_size)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr
