from cv2 import VideoCapture, resize, COLOR_BGR2GRAY, cvtColor, CascadeClassifier, rectangle, imshow, waitKey, destroyAllWindows, imread
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
import tensorflow as tf
# from skimage import transform

new_class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


emojis = [resize(imread(f'Emojis/{x}.png'), (500, 500)) for x in new_class_labels]
last_predictions = [4 for _ in range(20)]


vid = VideoCapture(0)
face_cascade = CascadeClassifier('../haarcascade_frontalface_default.xml')
my_model = load_model('../my_model_face_recognition_100_epochs.h5')
while vid.isOpened():
    ret, frame = vid.read()
    frame_bw = cvtColor(frame, COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_bw, 1.1, 4)
    for (x, y, w, h) in faces:
        rectangle(frame_bw, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = frame_bw[y:y + h, x:x + w]
    face = resize(face, (48, 48))
    face_r = np.expand_dims(face, axis=0)
    face_r = face_r / 255.0
    prediction = my_model.predict(face_r)
    predictions_temp = np.argmax(prediction, axis=1)
    prediction = np.argmax(prediction)
    last_predictions.append(prediction)
    del last_predictions[0]
    mean_prediction = max(set(last_predictions), key=last_predictions.count)

    imshow('VIDEOCAM', frame)
    imshow('EMOJI', emojis[mean_prediction])
    waitKey(1)
    # if waitKey(1) == 27:
    #     destroyAllWindows()


    # destroyAllWindows()

    # plt.imshow(frame, cmap='gray')
    # plt.title(prediction)
    # plt.show()
