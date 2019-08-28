from keras.models import load_model
import pickle
from keras.preprocessing import image
import numpy as np
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
class_labels= {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}


def face_detector(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
    except:
        return (x,w,y,h), np.zeros((48,48), np.uint8), img
    return (x,w,y,h), roi_gray, img


CLASSIFIER="XGB"

if CLASSIFIER=="XGB":
    classifier = pickle.load(open("../models/pkl_best_model_xgboost.pkl", "rb"))
else:
    classifier = load_model('../models/model_v6_23.hdf5')

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)
    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        # make a prediction on the ROI, then lookup the class
        if CLASSIFIER=="XGB":
            roi = roi.reshape((1,-1))
            #print(roi.shape)
            preds = classifier.predict(roi)[0]
            label = class_labels[int(preds)]
        else:
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
        #print(preds.argmax())

        label_position = (rect[0] + int((rect[1]/2)), rect[2] + 25)
        cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
    else:
        cv2.putText(image, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)

    cv2.imshow('All', image)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
