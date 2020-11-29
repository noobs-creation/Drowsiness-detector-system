import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leftEye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
rightEye = cv2.CascadeClassifier('haar cascade files\haarcascqade_righteye_2splits.xml')

label = ['Close', 'Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thickness = 2
rightEyePrediction = [99]
leftEyePrediction = [99]

while True:
    ret, frame = capture.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detectedFace = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    detectedLeftEye = leftEye.detectMultiScale(gray)
    detectedRightEye = rightEye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in detectedFace:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in detectedRightEye:
        right_Eye = frame[y:y + h, x:x + w]
        count = count + 1
        right_Eye = cv2.cvtColor(right_Eye, cv2.COLOR_BGR2GRAY)
        right_Eye = cv2.resize(right_Eye, (24, 24))
        right_Eye = right_Eye / 255
        right_Eye = right_Eye.reshape(24, 24, -1)
        right_Eye = np.expand_dims(right_Eye, axis=0)
        rightEyePrediction = model.predict_classes(right_Eye)
        if rightEyePrediction[0] == 1:
            label = 'Open'
        if rightEyePrediction[0] == 0:
            label = 'Closed'
        break

    for (x, y, w, h) in detectedLeftEye:
        left_Eye = frame[y:y + h, x:x + w]
        count = count + 1
        left_Eye = cv2.cvtColor(left_Eye, cv2.COLOR_BGR2GRAY)
        left_Eye = cv2.resize(left_Eye, (24, 24))
        left_Eye = left_Eye / 255
        left_Eye = left_Eye.reshape(24, 24, -1)
        left_Eye = np.expand_dims(left_Eye, axis=0)
        leftEyePrediction = model.predict_classes(left_Eye)
        if leftEyePrediction[0] == 1:
            label = 'Open'
        if leftEyePrediction[0] == 0:
            label = 'Closed'
        break

    if rightEyePrediction[0] == 0 and leftEyePrediction[0] == 0:
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    else:  # if(rightEyePrediction[0]==1 or leftEyePrediction[0]==1):
        score = score - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if score > 5:
        # person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:  # isplaying = False
            pass
        if thickness < 6:
            thickness = thickness + 2
        else:
            thickness = thickness - 2
            if thickness < 2:
                thickness = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thickness)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
