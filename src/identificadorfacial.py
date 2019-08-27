import os
import re
import numpy as np
import pickle
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sklearn import preprocessing
from sklearn import neighbors, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
import random

from sklearn.model_selection import GridSearchCV
from datetime import datetime
import xgboost as xgb


def separate_data(data_file_path):
    data_file = open(data_file_path)

    mylines = data_file.read().split('\n')
    #remove cabeçalho
    mylines.pop(0)
    #remove linha em branco do final
    mylines.pop(-1)

    #tipos=[]
    x=[]
    y=[]
    print (len(mylines))

    for line in mylines:
        vec=re.findall(r"[\w']+", line)
        y.append(vec.pop(0))
        x.append(vec)

    return np.array(x), np.array(y)


num_classes = 7
test_data_file_path = '../data/kaggle-data/fer2013/test.csv'
train_data_file_path = '../data/kaggle-data/fer2013/train.csv'
image_dir = "../images/"

emotion_dict= {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}

x_train, y_train = separate_data(train_data_file_path)
x_test, y_test = separate_data(test_data_file_path)

#print (x_train[0])
#print (y_train[0])
#
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#       rotation_range=30,
#       shear_range=0.3,
#       zoom_range=0.3,
#       horizontal_flip=True,
#       fill_mode='nearest')
#
# train_generator = train_datagen.flow(
#         x_train,
#         y_train,
#         save_to_dir = image_dir,
#         save_prefix = "train_"
#         target_size=(48,48),
#         color_mode="grayscale",
#         class_mode='categorical')



kn=10

kf = KFold(kn, shuffle=True)

Classifiers=["SVM", "RF","XGBoost"]

#this loop will interate for each classifier using diferents kfolds
for classifier in Classifiers:
    for train_index, val_index in kf.split(x_train):
        print ("Classifier: ",classifier)
        print ("using kfold= ",kn)
        print ("\n")
        #this will chose the classifier, and use gridSearch to choose the best hyper parameters focussing on reach the best AUC score
        #random forest
        if classifier == "RF":
          estimators = [ e for e in range(5, 25, 5) ]
          clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=dict(n_estimators=estimators), scoring="roc_auc", n_jobs=-1, cv=5, verbose=0)
        #XGBoost
        elif classifier == "XGBoost":
          clf = xgb.XGBClassifier()
        #SVM
        elif classifier == "SVM":
          Cs = [ 2.0**c for c in range(-5, 15, 1) ]
          Gs = [ 2.0**g for g in range(3, -15, -2) ]
          kernels = [ 'rbf', 'poly', 'sigmoid' ]
          decision_function_shapes = [ 'ovo', 'ovr' ]
          clf = GridSearchCV(estimator=SVC(probability=True), param_grid=dict(kernel=kernels, C=Cs, gamma=Gs, decision_function_shape=decision_function_shapes), scoring="roc_auc", n_jobs=-1, cv=5, verbose=0)

        #this fit will train your classifier to your dataset
        clf.fit(x_train[train_index],y_train[train_index])
        #this will return the probability of each example be on all classes
        #y_pred = clf.predict_proba(x_train[val_index])
        y_pred = clf.predict(x_train[val_index])

        #this will generate the AUC score if i use proba
        # auc = roc_auc_score(y[val_index], y_pred[:,1])
        # print "AUC: ", str(auc)
        # print "###########################\n"

        accuracy = accuracy_score(y_train[val_index], y_pred)
        print ("accuracy: ", str(accuracy))
        print ("###########################\n")

        #this will save the greater value, the estimator (model), the train and test set, to reproduce the best model with the real train set file
        if (classifier!="XGBoost"):
            if metrics[0]<auc:
                metrics=[accuracy, clf, clf.best_estimator_, x_train[train_index], y_train[train_index]]
        else:
            if metrics[0]<auc:
                metrics=[accuracy, clf, "xgboost",x_train[train_index], y_train[train_index]]

#save the best classificator into an file
pkl_filename = "../models/pkl_best_model_"+metrics[2]+".pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(metrics[1], file)


#he must use the classifier with the best score
if metrics[1]=="xgboost":
    clf_greater=xgb.XGBClassifier()
else:
    clf_greater=metrics[1]
#clf_greater = metrics[1]
clf_greater.fit(metrics[3], metrics[4])

test_y_pred = clf_greater.predict(x_test)
test_accuracy = accuracy_score(y_test, test_y_pred)

#generete logs, with statistics
print ("best estimator: ", metrics[2])
print ("greater accuracy in validation: ", metrics[0])
print("accuracy in test:", test_accuracy)
file_name="../results/statistics-"+metrics[2]+".txt"
with open(file_name, "a+") as f:
    f.write("###################################\n")
    f.write("\nbest estimator: "+ str(metrics[1]))
    f.write("\ngreater accuracy in validation: "+ str(metrics[0]))
    f.write("\naccuracy in test:", test_accuracy)
    f.write("\nconjunto de teste: "+ str(metrics[3]))
    f.write("\nconjunto de treino: "+ str(metrics[4]))
    f.write("\n###################################\n")
