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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Natural mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # EAR LANDMARKS (MediaPipe Index: 234 = left ear, 454 = right ear)
        left_ear = landmarks[234]
        right_ear = landmarks[454]

        # Convert to pixel coords
        lx, ly = int(left_ear.x * w), int(left_ear.y * h)
        rx, ry = int(right_ear.x * w), int(right_ear.y * h)

        # Draw gold glowing ellipses at ears
        overlay = frame.copy()

        for (x, y) in [(lx, ly), (rx, ry)]:
            cv2.ellipse(overlay, (x, y), (60, 80), 0, 0, 360, gilt_color, -1)

        # Blend with transparency
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        #Drawing ear points
        cv2.circle(frame, (lx, ly), 3, (0, 140, 255), -1)
        cv2.circle(frame, (rx, ry), 3, (0, 140, 255), -1)

        cv2.putText(frame, "Gilt FX Ears Active", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gilt_color, 2)

    cv2.imshow("Gold Ear FX", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()