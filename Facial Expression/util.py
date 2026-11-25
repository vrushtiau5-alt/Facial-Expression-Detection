import numpy as np
import cv2


EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def preprocess_face(face_img, target_size=(48, 48)):
    # Expects a BGR or grayscale face image from OpenCV
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_img, target_size)
    face_resized = face_resized.astype('float32') / 255.0
    face_resized = np.expand_dims(face_resized, -1)  # (48,48,1)
    face_resized = np.expand_dims(face_resized, 0)   # (1,48,48,1)
    return face_resized
