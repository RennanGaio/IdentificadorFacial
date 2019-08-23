import os
import re
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from PIL import Image
#import glob
#import cv2
#from sklearn.model_selection import train_test_split
#from keras.preprocessing.image import ImageDataGenerator
# from keras.layers import Dropout, Dense
# from keras.layers.normalization import BatchNormalization
# from keras.models import Sequential, load_model
# from keras.applications import VGG16
#from sklearn.metrics import accuracy_score, confusion_matrix


num_classes = 7
test_data_file = '../data/kaggle-data/fer2013/test.csv'
train_data_file = '../data/kaggle-data/fer2013/train.csv'

emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

file = open(train_data_file)

mylines = file.read().split('\n')
mylines.pop(0)

tipos=[]
count=0
data_vec=[]
print (len(mylines))
for line in mylines:
    vec=re.findall(r"[\w']+", line)
    data_vec.append(vec)
    if len(vec) < 1:
        continue
    if vec[0] not in tipos:
        tipos.append(line[-1])


print (tipos)
#
# val_datagen = ImageDataGenerator(rescale=1./255)
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#       rotation_range=30,
#       shear_range=0.3,
#       zoom_range=0.3,
#       horizontal_flip=True,
#       fill_mode='nearest')
# train_generator = train_datagen.flow_from_directory(
#         train_data_dir,
#         target_size=(48,48),
#         batch_size=batch_size,
#         color_mode="grayscale",
#         class_mode='categorical')
#
# validation_generator = val_datagen.flow_from_directory(
#         validation_data_dir,
#         target_size=(48,48),
#         batch_size=batch_size,
#         color_mode="grayscale",
#         class_mode='categorical')
