# emotion.py# emotion.py
# Real-time Emotion Detection using a pre-trained model and webcam input
import os
import cv2
import numpy as np
from keras.models import load_model

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the pre-trained emotion detection model
model = load_model(r"D:\vscode\real-time emotion detection\emotion_model.hdf5", compile=False)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define emotion labels corresponding to model output
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened successfully
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

# Start capturing video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
        # If frame is not captured, exit the loop
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64)) # Resize to match model input
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)    
        roi_gray = np.expand_dims(roi_gray, axis=-1)  

        # Predict emotion
        prediction = model.predict(roi_gray, verbose=0)[0]
        emotion = emotion_labels[np.argmax(prediction)]

        # Draw face rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)

    cv2.imshow("Real-time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
