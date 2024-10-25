import cv2
import numpy as np
import os
import csv
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
    
with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgbackground = cv2.imread("imgback.jpg")

COL_NAMES = ['NAME', 'TIME']

if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
csv_filename = f"Attendance/Attendance_{date}.csv"

if not os.path.isfile(csv_filename):
    with open(csv_filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(COL_NAMES)

while True:
    ret, frame = video.read()
    
    if not ret:
        print("Failed to capture image from webcam. Exiting...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, dsize=(100, 100)).flatten().reshape(1, -1)
        
        try:
            output = knn.predict(resized_img)
        except ValueError as e:
            print(f"Prediction error: {e}")
            continue
        
        timestamp = datetime.now().strftime("%H:%M:%S")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        attendance = [str(output[0]), timestamp]
        imgbackground[162:162+480, 55:55+640] = frame

    cv2.imshow("Frame", imgbackground)
    
    k = cv2.waitKey(1)

    if k == ord('k'):
        with open(csv_filename, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(attendance)
        time.sleep(5)
    if k == 32:
        break

video.release()
cv2.destroyAllWindows()
