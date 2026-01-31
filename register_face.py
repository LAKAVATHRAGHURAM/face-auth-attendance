import cv2
import os
import numpy as np

user = input("Enter user name: ")
os.makedirs("dataset", exist_ok=True)
user_path = f"dataset/{user}"
os.makedirs(user_path, exist_ok=True)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0
while count < 30:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        cv2.imwrite(f"{user_path}/{count}.jpg", face)
        count += 1

    cv2.imshow("Register Face", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Face registered successfully")
