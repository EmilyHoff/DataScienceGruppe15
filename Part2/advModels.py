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
#from tensorflow.keras.models import Sequential
#import fasttext
#from tensorflow.keras import metrics
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

from imblearn.under_sampling import RandomUnderSampler


#sys.path.insert(0,"../")

#import main
import wandb

#df = None

#from transformers import pipeline
import os

def bert(encoded,embedding_matrix,labels,vocab_size,split):

    trainX = tf.convert_to_tensor(encoded[:split],dtype=tf.float32)
    trainY = tf.convert_to_tensor(labels[:split])
    
    testX = tf.convert_to_tensor(encoded[split:],dtype=tf.float32)
    testY = tf.convert_to_tensor(labels[split:])    
    
    print(f"Lenght of train: {trainX}")
    print(f"Length of train y: {trainY}")
    
    
    
    batch_size = 32
    
    print()
    
    #RNN
    
    bertModel = Sequential()
    bertModel.add(tf.keras.layers.Embedding(vocab_size+1,150,weights=[embedding_matrix]))
    bertModel.add(tf.keras.layers.LSTM(32))
    bertModel.add(tf.keras.layers.GaussianDropout(0.3))
    bertModel.add(tf.keras.layers.Reshape((2,16)))
    bertModel.add(tf.keras.layers.LSTM(16))
    bertModel.add(tf.keras.layers.GaussianDropout(0.2))
    bertModel.add(tf.keras.layers.Reshape((2,8)))
    bertModel.add(tf.keras.layers.LSTM(8))
    bertModel.add(tf.keras.layers.Dense(1,activation="sigmoid"))
    '''
    
    #Proven to work
    bertModel = Sequential()
    bertModel.add(tf.keras.layers.Embedding(vocab_size+1,150,weights=[embedding_matrix]))
    bertModel.add(tf.keras.layers.Dense(32))
    bertModel.add(tf.keras.layers.Dropout(0.3))
    bertModel.add(tf.keras.layers.Dense(16))
    bertModel.add(tf.keras.layers.Dense(1))
    
    #Dense
    
    bertModel = Sequential()
    bertModel.add(tf.keras.layers.Embedding(vocab_size+1,150,weights=[embedding_matrix]))
    bertModel.add(tf.keras.layers.Dense(32))
    bertModel.add(tf.keras.layers.GaussianDropout(0.2))
    bertModel.add(tf.keras.layers.Dense(16))
    bertModel.add(tf.keras.layers.GaussianDropout(0.2))
    bertModel.add(tf.keras.layers.Dense(8))
    bertModel.add(tf.keras.layers.Dense(1,activation="sigmoid"))
    '''
    
    METRICS = [
      metrics.TruePositives(name='tp'),
      metrics.FalsePositives(name='fp'),
      metrics.TrueNegatives(name='tn'),
      metrics.FalseNegatives(name='fn'), 
      metrics.BinaryAccuracy(name='accuracy'),
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall'),
      metrics.AUC(name='auc'),
      metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    
    bertModel.compile(optimizer='adam',loss='binary_crossentropy',metrics=[METRICS])
    
    
    for i in range(len(trainX) // batch_size):
        print(f"batch: {i}")
        x_batch = trainX[i*batch_size:(i+1)*batch_size]
        y_batch = trainY[i*batch_size:(i+1)*batch_size]
        bertModel.fit(x_batch, y_batch)
        
    print(bertModel.predict(testX))
    loss, accuracy,auc,precision,recall = bertModel.evaluate(testX,testY)
    
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    
    return loss,accuracy

def ensemble(encoded,labels,split):
    
    trainX = np.array(encoded[:split])
    trainY = np.array(labels[:split])
    
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(trainX)
    
    smote = SMOTE()
    trainX,trainY = smote.fit_resample(X_train_scaled,trainY)


    testX = np.array(encoded[split:])
    testY = np.array(labels[split:])
        
    #clf = BaggingClassifier(base_estimator=LogisticRegression(random_state=0,max_iter=15,solver="newton-cg"),
    #                        n_estimators=25,
    #                        random_state=0).fit(trainX,trainY)
    
    
    #clf = BaggingClassifier(base_estimator=svm.SVC(),
    #                       n_estimators=15,
    #                       random_state=0,verbose=1).fit(trainX,trainY)
    
    #clf = BaggingClassifier(base_estimator=KMeans(n_clusters=2, n_init="auto"),
    #                        n_estimators=25,
    #                        random_state=0).fit(trainX,trainY)
    
    
    #clf = RandomForestClassifier(n_estimators=25,class_weight="balanced").fit(trainX,trainY)
    
    #clf = ExtraTreesClassifier(n_estimators=250, max_depth=5,
    #                           min_samples_split=2, random_state=0,
    #                           verbose=4).fit(trainX,trainY)
    
    clf = AdaBoostClassifier(base_estimator=LogisticRegression(random_state=0,max_iter=15,solver="newton-cg"),
                             n_estimators=300).fit(trainX,trainY)
    
    y_pred = clf.predict(testX)
    print(f"This is the ensemble method: {classification_report(testY,y_pred)}")
    return y_pred








