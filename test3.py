import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Start webcam
cap = cv2.VideoCapture(0)

# Gilt color (gold)
gilt_color = (0, 215, 255)

def is_headphone_on(patch):
    """Estimate if headphones are on by checking average darkness."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < 70  # adjust threshold if needed

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

        # EAR landmark indexes
        left_ear = landmarks[234]
        right_ear = landmarks[454]

        lx, ly = int(left_ear.x * w), int(left_ear.y * h)
        rx, ry = int(right_ear.x * w), int(right_ear.y * h)

        show_fx = False

        try:
            # Crop 60x60 patch around both ears
            patch_left = frame[ly - 30:ly + 30, lx - 30:lx + 30]
            patch_right = frame[ry - 30:ry + 30, rx - 30:rx + 30]

            left_on = is_headphone_on(patch_left)
            right_on = is_headphone_on(patch_right)

            if left_on or right_on:
                show_fx = True

        except:
            # Ear region may be off screen
            pass

        if show_fx:
            # Draw glowing FX around ears
            overlay = frame.copy()
            for (x, y) in [(lx, ly), (rx, ry)]:
                cv2.ellipse(overlay, (x, y), (60, 80), 0, 0, 360, gilt_color, -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            cv2.putText(frame, "Headphones Detected ", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gilt_color, 2)
        else:
            cv2.putText(frame, "No Headphones", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Optional: Draw ear points
        cv2.circle(frame, (lx, ly), 3, (0, 140, 255), -1)
        cv2.circle(frame, (rx, ry), 3, (0, 140, 255), -1)

    cv2.imshow("Gold FX with Headphone Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()