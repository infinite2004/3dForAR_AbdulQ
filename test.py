import cv2
import numpy as np
import requests
import time
from spotipy.oauth2 import SpotifyOAuth
import spotipy
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# === Spotify Setup ===
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="",
    client_secret="",
    redirect_uri="http://localhost:8888/callback/",
    scope="user-read-currently-playing"
))

# === Teachable Machine Model Setup ===
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def detect_headphones(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img).resize((224, 224))
    pil_img = ImageOps.grayscale(pil_img) if pil_img.mode != "RGB" else pil_img

    img_array = np.asarray(pil_img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]

    print(f"Prediction: {class_names[class_idx].strip()} ({confidence:.2f})")
    return class_idx == 0 and confidence > 0.8  # Assuming class 0 = "Headphones On"

# === Album Cover & Scroll Config ===
last_album_url = None
album_image = None
scroll_offset = 0
scroll_speed = 2
overlay_width = 120
overlay_height = 120

# === Start Camera ===
cap = cv2.VideoCapture(0)

def get_album_cover():
    current = sp.current_playback()
    if current and current['is_playing']:
        return current['item']['album']['images'][0]['url']
    return None

def fetch_album_image(url):
    resp = requests.get(url)
    arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.resize(img, (240, overlay_height))  # Wider for scrolling

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run headphone detection using model
    headphones_on = detect_headphones(frame)

    # Fetch album art if song changed
    album_url = get_album_cover()
    if album_url and album_url != last_album_url:
        album_image = fetch_album_image(album_url)
        last_album_url = album_url
        scroll_offset = 0

    # Overlay scrolling image only if headphones are on
    if headphones_on and album_image is not None:
        scroll_offset = (scroll_offset + scroll_speed) % (album_image.shape[1] - overlay_width)
        scroll_slice = album_image[0:overlay_height, scroll_offset:scroll_offset + overlay_width]
        x, y = 100, 100
        frame[y:y+overlay_height, x:x+overlay_width] = scroll_slice

    cv2.imshow("AR Spotify Headphones", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()