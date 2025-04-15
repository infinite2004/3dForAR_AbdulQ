import cv2
import numpy as np
import mediapipe as mp
import os
import time
import requests
from PIL import Image
from io import BytesIO
from spotipy.oauth2 import SpotifyOAuth
import spotipy
import threading

# ====== Spotify Setup ======
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="",
    client_secret="",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-read-playback-state user-read-currently-playing"
))

def save_album_art():
    while True:
        try:
            current = sp.current_playback()
            if current and current['item']:
                url = current['item']['album']['images'][0]['url']
                response = requests.get(url)
                img_pil = Image.open(BytesIO(response.content)).resize((60, 60))
                img_pil.save("current_album.jpg")
        except Exception as e:
            print("Spotify Fetch Error:", e)
        time.sleep(5)

# ====== Perspective Tilt Function ======
def tilt_image(img, direction="x", tilt=5):
    h, w = img.shape[:2]

    # Original corner points
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    if direction == "x":
        # Tilt around X-axis (vertical tilt)
        dst_pts = np.float32([
            [0, h * tilt],
            [w, 0],
            [w, h],
            [0, h - h * tilt]
        ])
    elif direction == "y":
        # Tilt around Y-axis (horizontal skew)
        dst_pts = np.float32([
            [w * tilt, 0],
            [w - w * tilt, 0],
            [w, h],
            [0, h]
        ])
    else:
        return img

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return warped

# ====== MediaPipe Setup ======
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

gilt_color = (0, 215, 255)

def is_headphone_on(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < 70

def load_album_image():
    if os.path.exists("current_album.jpg"):
        try:
            img = cv2.imread("current_album.jpg")
            return img
        except:
            return None
    return None

# ====== Start Album Fetcher Thread ======
threading.Thread(target=save_album_art, daemon=True).start()

# ====== Start Camera and Overlay ======
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = landmarks[234]
        right_ear = landmarks[454]
        lx, ly = int(left_ear.x * w), int(left_ear.y * h)
        rx, ry = int(right_ear.x * w), int(right_ear.y * h)

        show_fx = False
        try:
            patch_left = frame[ly - 30:ly + 30, lx - 30:lx + 30]
            patch_right = frame[ry - 30:ry + 30, rx - 30:rx + 30]
            left_on = is_headphone_on(patch_left)
            right_on = is_headphone_on(patch_right)
            if left_on or right_on:
                show_fx = True
        except:
            pass

        if show_fx:
            album_img = load_album_image()
            if album_img is not None:
                img_size = 150
                album_img = cv2.resize(album_img, (img_size, img_size))

                positions = [("left", lx, ly), ("right", rx, ry)]

                for side, x, y in positions:
                    try:
                        half = img_size // 2

                        # Custom offsets and 3D tilt
                        if side == "left":
                            x_offset = -70
                            y_offset = -20
                            transformed_img = tilt_image(album_img, direction="x", tilt=0.)
                        else:
                            x_offset = 60
                            y_offset = -20
                            transformed_img = tilt_image(album_img, direction="y", tilt=0.3)

                        # Overlay image
                        x1, y1 = x - half + x_offset, y - half + y_offset
                        x2, y2 = x + half + x_offset, y + half + y_offset

                        if 0 <= x1 < w and 0 <= y1 < h and x2 <= w and y2 <= h:
                            frame[y1:y2, x1:x2] = transformed_img
                    except:
                        pass

            cv2.putText(frame, "Now Playing", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gilt_color, 2)
        else:
            cv2.putText(frame, "No Headphones", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.circle(frame, (lx, ly), 3, (0, 140, 255), -1)
        cv2.circle(frame, (rx, ry), 3, (0, 140, 255), -1)

    cv2.imshow("Headphone Album Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
