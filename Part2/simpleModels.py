from base64 import encode
from mimetypes import init
from multiprocessing.dummy import active_children
import os
import shutil
from tkinter import E
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import sys
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
import fasttext
from tensorflow.keras import metrics
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

def perceptron(encoded,labels,split):
    
    scaler = StandardScaler()
    
    trainX = encoded[:split]
    trainX = np.array(trainX)
    trainX = trainX.reshape((trainX.shape[0], -1))
    trainX = scaler.fit_transform(trainX)
    
    trainY = np.array(labels[:split])
    
    testX = encoded[split:]
    testX = np.array(testX)
    testX = testX.reshape((testX.shape[0], -1))
    testX = scaler.transform(testX)
    
    testY = np.array(labels[split:])
    
    clf = Perceptron(tol=1e-3, random_state=0,max_iter=15)
    clf.fit(trainX,trainY)
    
    print(clf.score(testX,testY))
    
    
def log(encoded,labels,split):
    scaler = StandardScaler()
    
    trainX = encoded[:split]
    trainX = np.array(trainX)
    trainX = trainX.reshape((trainX.shape[0], -1))
    trainX = scaler.fit_transform(trainX)
    
    trainY = np.array(labels[:split])
    
    testX = encoded[split:]
    testX = np.array(testX)
    testX = testX.reshape((testX.shape[0], -1))
    testX = scaler.transform(testX)
    
    testY = np.array(labels[split:])
    
    clf = LogisticRegression(random_state=0,max_iter=15,solver="newton-cg")
    clf.fit(trainX, trainY)
    
    print(clf.score(testX,testY))
    
    
    
