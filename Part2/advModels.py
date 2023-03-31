from mimetypes import init
from multiprocessing.dummy import active_children
import os
import shutil
from tkinter import E
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sys
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
import fasttext
from tensorflow.keras import metrics

import wandb

import os

def LSTM(encoded,labels,split,vocab_size=None,embedding_matrix=None):

    trainX = tf.convert_to_tensor(encoded,dtype=tf.float32)
    trainY = tf.convert_to_tensor(labels)
    
    testX = tf.convert_to_tensor(encoded[split:],dtype=tf.float32)
    testY = tf.convert_to_tensor(labels[split:])    
    
    print(f"Lenght of train: {trainX}")
    print(f"Length of train y: {trainY}")

    batch_size = 32

    LSTMModel = Sequential()
    LSTMModel.add(tf.keras.layers.Dense(128, activation="relu"))
    LSTMModel.add(tf.keras.layers.Reshape((2,64)))
    LSTMModel.add(tf.keras.layers.GaussianDropout(0.2))
    LSTMModel.add(tf.keras.layers.LSTM(128, activation="tanh"))
    LSTMModel.add(tf.keras.layers.GaussianDropout(0.2))
    LSTMModel.add(tf.keras.layers.Reshape((2*64,)))
    LSTMModel.add(tf.keras.layers.Dense(16, activation='relu'))
    LSTMModel.add(tf.keras.layers.Reshape((2, 8)))
    LSTMModel.add(tf.keras.layers.LSTM(8))
    LSTMModel.add(tf.keras.layers.Dense(1, activation="sigmoid",
                                        bias_initializer=tf.keras.initializers.Constant(np.log([sum(labels)/len(labels)]))))

    METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve,
    ]
    
    LSTMModel.compile(optimizer='adam',loss='binary_crossentropy',metrics=METRICS)
    
    for i in range(len(trainX) // batch_size):
        print(f"batch: {i}")
        x_batch = trainX[i*batch_size:(i+1)*batch_size]
        y_batch = trainY[i*batch_size:(i+1)*batch_size]
        LSTMModel.fit(x_batch, y_batch,class_weight={0:1,1:1})
        
    metric = LSTMModel.evaluate(testX,testY)
    return metric









