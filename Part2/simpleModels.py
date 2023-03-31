from base64 import encode
from mimetypes import init
from multiprocessing.dummy import active_children
import os
import shutil
from tkinter import E
import numpy as np
import sys
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn.model_selection as sk
import numpy as np
import re

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

def bow_logreg(encoded, labels, split):
    '''A logistical regression model for binary classification of fake news
    A baseline, only trained on the content of the article, the content is 
    encoded using the bag-of-words scheme'''

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

    return y_pred, logreg

def numberOfAuthors(df):
    '''Drops the current authors column and replaces it with a new column containing the
    number of authers credited for the publication'''
    authors = df['authors'].tolist()
    for x in range(0, len(df)):
        nanRes = len(re.findall(r"nan\b", str(authors[x])))
        if nanRes == 0: #there are authors
            result = len(re.findall(r",", str(authors[x]))) 
            authors[x] = result + 1
        else:
            authors[x] = 0
    return pd.DataFrame(authors, columns=['authors'])

def predictByAuthors(df, labels, split):
    '''A logistical baseline model that predicts the label of an article only based on
    the number of authers who wrote it'''
    df = numberOfAuthors(df)

    authors = np.array(df)
    labels = np.array(labels)

    x_train = authors[:split]
    y_train = labels[:split]
    x_val = authors[split:]
    y_val =labels[split:]

    authorMod = LogisticRegression()

    authorMod = authorMod.fit(x_train, y_train)

    y_pred = authorMod.predict(x_val)

    print("Accuracy: ", metrics.accuracy_score(y_val, y_pred))
    print("Recall: ", metrics.recall_score(y_val, y_pred, average='weighted', zero_division=1))
    print("Precision: ", metrics.recall_score(y_val, y_pred, average='weighted', zero_division=1))
    print("F1-score: ", metrics.f1_score(y_val, y_pred, average='weighted', zero_division=1))

    return y_pred, authorMod