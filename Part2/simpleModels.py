from base64 import encode
from mimetypes import init
from multiprocessing.dummy import active_children
import os
import shutil
from tkinter import E
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
#import tensorflow_text
import sys
from sklearn.metrics import accuracy_score
'''from tensorflow.keras.models import Sequential
import fasttext
from tensorflow.keras import metrics'''
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve
import sklearn.model_selection as sk
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

def bow_perceptron(encoded, labels, split):

    articles = np.array(encoded)
    labels = np.array(labels)

    X_train = articles[:split]
    y_train = labels[:split]
    X_val = articles[split:]
    y_val = labels[split:]

    percep = Perceptron()
    percep.fit(X_train, y_train)
    y_pred = percep.predict(X_val)

    accuracy = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred, zero_division=1,average='weighted')
    recall = metrics.recall_score(y_val, y_pred, zero_division=1,average='weighted')
    f1_score = metrics.f1_score(y_val, y_pred, zero_division=1, average='weighted')
    conf_matr = metrics.confusion_matrix(y_val, y_pred)
    display_confMatr = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matr,
                                                      display_labels=[False, True])
    class_report = metrics.classification_report(y_val, y_pred, zero_division=1)

    print('percep accuracy: {:0.5f}'.format(accuracy))
    print('percep precision: {:0.5f}'.format(precision))
    print('percep recall: {:0.5f}'.format(recall))
    print('percep f1-score: {:0.5f}'.format(f1_score))
    print(class_report)
    # display_confMatr.plot()
    # plt.show()
    return y_pred

def bow_logreg(encoded, labels, split):

    articles = np.array(encoded)
    labels = np.array(labels)

    X_train = articles[:split]
    y_train = labels[:split]
    X_val = articles[split:]
    y_val = labels[split:]

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_val)

    accuracy = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred, zero_division=1,average='weighted')
    recall = metrics.recall_score(y_val, y_pred, zero_division=1,average='weighted')
    f1_score = metrics.f1_score(y_val, y_pred, zero_division=1, average='weighted')
    conf_matr = metrics.confusion_matrix(y_val, y_pred)
    class_report = metrics.classification_report(y_val, y_pred, zero_division=1)

    print('logreg accuracy: {:0.5f}'.format(accuracy))
    print('logreg precision: {:0.5f}'.format(precision))
    print('logreg recall: {:0.5f}'.format(recall))
    print('logreg f1-score: {:0.5f}'.format(f1_score))
    print('logreg confusion matrix:\n', conf_matr,'\n')
    print(class_report)

    return y_pred

def numberOfAuthors(df):
    authors = df['authors'].tolist()
    for x in range(0, len(df)):
        if type(authors[x]) == float:
            df['authors'][x] = 0
        else:
            result = re.findall(r",", str(authors[x]))
            df['authors'][x] = len(result) + 1
    return df

def predictByAuthors(trainDf, valDf):
    #prepare data
    trainDf = numberOfAuthors(trainDf)
    valDF = numberOfAuthors(valDf)

    x_train = trainDf['authors'].to_numpy()
    y_train = trainDf['type'].to_numpy()

    x_val = valDf['authors'].to_numpy()
    y_val = valDf['type'].to_numpy()

    authorMod = LogisticRegression()

    #reshaping x_data
    x_train = np.reshape(x_train, (-1, 1))
    x_val = np.reshape(x_val, (-1, 1))

    #ensuring correct type within y_data
    y_train = y_train.astype('int')
    y_val = y_val.astype('int')

    authorMod = authorMod.fit(x_train, y_train)

    y_pred = authorMod.predict(x_val)
    y_val = y_val.astype('int')

    print("Accuracy: ", metrics.accuracy_score(y_val, y_pred))
    print("Recall: ", metrics.recall_score(y_val, y_pred, average='weighted', zero_division=1))
    print("Precision: ", metrics.recall_score(y_val, y_pred, average='weighted', zero_division=1))
    print("F1-score: ", metrics.f1_score(y_val, y_pred, average='weighted', zero_division=1))

    #confusion matrix
    confusionMatrix = metrics.confusion_matrix(y_val, y_pred, normalize='all')
    cmTable = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,
                                             display_labels=[False, True])
    cmTable.plot()
    plt.show()

    return y_pred

def ROCcurve(y_predA, y_predLog, y_predPer, y_predEns, valDf):
    #prepare data
    y_val = valDf['type'].to_numpy()
    y_val = y_val.astype('int')

    #auc on author model
    fpr, tpr, _ = metrics.roc_curve(y_val,  y_predA)
    aucA = metrics.roc_auc_score(y_val, y_predA)
    plt.plot(fpr, tpr, label="Author auc= " + str(aucA))

    #auc on logMod
    fpr, tpr, _ = metrics.roc_curve(y_val,  y_predLog)
    aucLog = metrics.roc_auc_score(y_val, y_predLog)
    plt.plot(fpr, tpr, label="LogMod auc= " + str(aucLog))

    #auc on perceptron
    fpr, tpr, _ = metrics.roc_curve(y_val,  y_predPer)
    aucPer = metrics.roc_auc_score(y_val, y_predPer)
    plt.plot(fpr, tpr, label="Perceptron auc= " + str(aucPer))

    #auc on ensemble
    fpr, tpr, _ = metrics.roc_curve(y_val,  y_predEns)
    aucEns = metrics.roc_auc_score(y_val, y_predEns)
    plt.plot(fpr, tpr, label="Perceptron auc= " + str(aucEns))

    plt.legend(loc=4)
    plt.title("ROC-curves")
    plt.show()
    return