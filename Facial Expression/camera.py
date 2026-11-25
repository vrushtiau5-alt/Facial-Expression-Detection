import cv2
import numpy as np
from tensorflow.keras.models import load_model
from util import preprocess_face, EMOTIONS


def main(model_path='model.h5', camera_idx=0):
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(face_cascade_path)

    model = load_model(model_path)

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise SystemExit('Could not open webcam')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            inp = preprocess_face(face)
            preds = model.predict(inp)
            label = EMOTIONS[int(np.argmax(preds))]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Real-time facial emotion recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
