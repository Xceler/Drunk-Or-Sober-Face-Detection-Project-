# Real-Time Drunk or Sober Detection

This project uses a webcam to detect faces in real-time and classify each detected face as 'Drunk' or 'Sober' using a pre-trained deep learning model.

## Project Overview

The project captures video from the webcam, detects faces in each frame, preprocesses the detected face images, and then uses a machine learning model to classify each face as 'Drunk' or 'Sober'. The results are displayed on the video feed in real-time.

## Requirements

- Python 3.x
- OpenCV
- TensorFlow/Keras
- NumPy

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Xceler/drunk-or-sober-detection.git
    cd drunk-or-sober-detection
    ```

2. **Install Required Libraries:**

    ```bash
    pip install opencv-python tensorflow numpy
    ```

## Usage

1. **Download or Train a Model:**
    - If you have a pre-trained model, place it in the project directory and update the model path in the script.
    - If you need to train a model, follow a suitable tutorial to train a model on a dataset that differentiates between drunk and sober faces.

2. **Run the Script:**

    ```bash
    python real_time_detection.py
    ```

## Project Structure

- `real_time_detection.py`: Main script to run the real-time detection.
- `README.md`: This file.

## real_time_detection.py

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_face(face):
    face = cv2.resize(face, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def predict(face, model):
    face = preprocess_face(face)
    prediction = model.predict(face)
    return 'Drunk' if prediction[0][0] <= 0.5 else 'Sober'

def real_time_detection(model):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            label = predict(face, model)

            color = (0, 255, 0) if label == 'Sober' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Real-Time Drunk Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load your pre-trained model
model = load_model('path_to_model.h5')

# Start real-time detection
real_time_detection(model)
