# Face Detection Attendance System

This project implements a face detection-based attendance system using Python, OpenCV, and machine learning. The system captures the face of a student in real-time, recognizes it, and marks their attendance in a CSV file.

## Features
- Real-time face detection using OpenCV.
- Machine learning with K-Nearest Neighbors (KNN) for face recognition.
- Attendance records stored in CSV format for easy access and review.
- Automatic timestamp generation for attendance marking.

## Technologies Used
- **Python**
- **OpenCV (cv2)**: For face detection.
- **Numpy**: To handle arrays.
- **CSV**: To store attendance in a spreadsheet-like format.
- **Datetime**: For timestamping attendance records.
- **OS**: For handling file directories.
- **Pickle**: For saving and loading models and data efficiently.

## Files
- **face_data.pkl**: Stores the face data used for training the recognition model.
- **names.pkl**: Stores the corresponding names of the faces.
- **haarcascade_frontalface_default.xml**: Pre-trained XML file for detecting faces using OpenCV.
- **Attendance_<date>.csv**: File where the attendance records (Name and Timestamp) are stored for each session.

## How It Works
1. **Face Data Collection**: Run the `collect_faces.py` script to capture a specified number of face samples from the webcam, resize them, and store the data in `data/face_data.pkl`. Corresponding names are saved in `data/names.pkl`.
2. **Face Recognition and Attendance**: Run the `attendance.py` script, which loads the stored face data and applies the KNN algorithm to detect and recognize faces in real-time using the webcam. When a face is recognized, the system logs the student's name and the timestamp in a CSV file for attendance.
3. **CSV Output**: Attendance records are saved in a CSV file with the format `Attendance_<current_date>.csv`. Each row contains the student's name and the time they were recognized.

## Requirements
- Python 3.x
- OpenCV
- Numpy
- Scikit-learn
- Pickle
- A webcam

## Usage
1. Install the required packages.
   ```bash
   pip install opencv-python numpy scikit-learn
