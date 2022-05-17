from cv2 import VideoCapture, resize, COLOR_BGR2GRAY, cvtColor, CascadeClassifier, rectangle
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
import tensorflow as tf
# from skimage import transform


vid = VideoCapture(0)

while vid.isOpened():
    ret, frame = vid.read()
    frame = cvtColor(frame, COLOR_BGR2GRAY)
    face_cascade = CascadeClassifier('../haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    for (x, y, w, h) in faces:
        rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = frame[y:y + h, x:x + w]
    face = resize(face, (48, 48))
    my_model = load_model('../my_model_face_recognition_100_epochs.h5')
    # face_r = resize(face, (48, 48))
    face_r = np.expand_dims(face, axis=0)
    face_r = face_r / 255.0
    prediction = my_model.predict(face_r)
    predictions_temp = np.argmax(prediction, axis=1)
    new_class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    prediction = np.argmax(prediction)
    prediction = new_class_labels[prediction]
    plt.imshow(face, cmap='gray')
    plt.title(prediction)
    plt.show()
