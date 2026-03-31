import cv2
import numpy as np

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Split face into upper and lower half
        lower_half = face[int(h/2):h, :]

        # Check brightness (simple trick)
        avg_brightness = np.mean(lower_half)

        if avg_brightness < 80:
            label = "Mask 😷"
            color = (0, 255, 0)
        else:
            label = "No Mask ❌"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
