import tensorflow as tf 
import cv2 
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 


# Data Preprocessing 

data_project = 'Project'

datagen = ImageDataGenerator(
    rescale = 1./255.0,
    width_shift_range =0.2,
    height_shift_range = 0.2,
    rotation_range = 15,
    shear_range = 0.2,
    zoom_range = 0.5,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

train_gen = datagen.flow_from_directory(
    data_project,
    batch_size = 32,
    target_size = (224, 224),
    class_mode = 'binary'
)


# Modeling With CNN Model 


Model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (224, 224, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),


    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(rate= 0.5),

    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(1, activation = 'sigmoid')
])


Model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

Model.fit(train_gen, 
         epochs = 20)

from collections import deque

# Face Detection and Classification about 'Drunk' Or 'Sober'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



# Function to preprocess and classify a face as drunk or sober
def classify_face(face):
    # Preprocess the face image
    face = cv2.resize(face, (224, 224))
    face = np.expand_dims(face, axis=0)
    face = face / 255.0  # Normalize

    # Predict using the model
    prediction = Model.predict(face)
    
    # Return classification label
    return "Sober" if prediction > 0.6 else "Drunk"  # Adjust threshold as needed

# Function to augment the face image with random transformations
def augment_face(face):
    # Define range for random rotation (-15 to 15 degrees)
    angle = np.random.randint(-15, 16)

    # Get the center of the image
    center = (face.shape[1] // 2, face.shape[0] // 2)

    # Define rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform rotation
    rotated_face = cv2.warpAffine(face, rotation_matrix, (face.shape[1], face.shape[0]))

    return rotated_face

# Open the default camera
cap = cv2.VideoCapture(0)

# Queue to store predictions for the last few frames
prediction_queue = deque(maxlen=5)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Reset prediction queue for each frame
    prediction_queue.clear()

    # Classify each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Augment the face image with random transformations
        augmented_face = augment_face(face_roi)

        # Classify the augmented face as drunk or sober
        label = classify_face(augmented_face)

        # Store the prediction in the queue
        prediction_queue.append(label)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display label on the frame
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # If prediction_queue is not empty, find the label with the highest count
    if prediction_queue:
        # Count the occurrences of each label in the queue
        counts = {label: prediction_queue.count(label) for label in set(prediction_queue)}

        # Get the label with the highest count
        final_label = max(counts, key=counts.get)

        # Display the final label on the frame
        cv2.putText(frame, final_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Drunk or Sober Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()2
