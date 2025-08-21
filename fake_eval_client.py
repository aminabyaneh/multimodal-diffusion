import cv2
import base64
import requests

import numpy as np

from PIL import Image
from scipy.spatial.transform import Rotation as R

# Config
SERVER_URL = "http://localhost:8777"
INSTRUCTION = "Slide the red cylinder across the table."


def decode_numpy_json(obj):
    if isinstance(obj, dict) and "__numpy__" in obj:
        array_bytes = base64.b64decode(obj["__numpy__"])
        return np.frombuffer(array_bytes, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    elif isinstance(obj, list):
        return np.array([decode_numpy_json(x) for x in obj])
    else:
        return obj


def crop_resize(img, output_size=(224, 224)):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    width, height = img.size
    min_dim = min(width, height)

    # Center crop box
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size, Image.BILINEAR)

    return np.array(img)


def get_observation():
    obs = {}
    color_image = np.zeros((224, 224, 3), dtype=np.uint8)

    # --- Resize the image to 224×224 using bilinear ---
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    resized_array = np.array(color_image).astype(np.uint8)
    obs["full_image"] = resized_array.tolist()
    obs["eef_pos"] = [0.0, 0.0, 0.0]
    return obs


# Send to OpenVLA server
def get_vla_action(observation):
    response = requests.post(f"{SERVER_URL}/act", json=observation)

    action_raw = response.json()

    if response.status_code != 200:
        print("Error:", response.text)
        return None

    return decode_numpy_json(action_raw)


# === MAIN LOOP ===
if __name__ == "__main__":

    while True:
        observation = get_observation()
        print("Sending observation")
        observation["instruction"] = INSTRUCTION
        action = get_vla_action(observation)
        if action is not None:
            print("Received action:", action)
