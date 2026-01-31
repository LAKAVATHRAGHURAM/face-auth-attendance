import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

# ------------------ Face Recognition Setup ------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = []
labels = []
label_map = {}
label_id = 0

for user in os.listdir("dataset"):
    label_map[label_id] = user
    user_path = os.path.join("dataset", user)
    for img in os.listdir(user_path):
        img_path = os.path.join(user_path, img)
        faces.append(cv2.imread(img_path, 0))
        labels.append(label_id)
    label_id += 1

recognizer.train(faces, np.array(labels))

# ------------------ Attendance File ------------------
if os.path.exists("attendance.csv"):
    df = pd.read_csv("attendance.csv")
else:
    df = pd.DataFrame(columns=["Name", "Date", "Punch In", "Punch Out"])

# ------------------ Camera ------------------
cap = cv2.VideoCapture(0)
print("Look at the camera... Capturing once")

captured = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        label, confidence = recognizer.predict(face)

        if confidence < 70:
            name = label_map[label]
            today = str(datetime.now().date())
            time_now = datetime.now().strftime("%H:%M:%S")

            # Get today's records for this person
            today_records = df[(df["Name"] == name) & (df["Date"] == today)]

            if today_records.empty:
                # No record today → Punch In
                df.loc[len(df)] = [name, today, time_now, ""]
                print(name, "Punch In")

            else:
                # Always check LAST row only
                last_index = today_records.index[-1]
                last_row = df.loc[last_index]

                if pd.isna(last_row["Punch Out"]) or last_row["Punch Out"] == "":
                    # Fill Punch Out in SAME row
                    df.loc[last_index, "Punch Out"] = time_now
                    print(name, "Punch Out")
                else:
                    # Start new session → Punch In
                    df.loc[len(df)] = [name, today, time_now, ""]
                    print(name, "Punch In")

            df.to_csv("attendance.csv", index=False)
            captured = True
            break

    cv2.imshow("Attendance System", frame)
    cv2.waitKey(300)

    if captured:
        break

cap.release()
cv2.destroyAllWindows()
print("Attendance updated. Program closed.")
