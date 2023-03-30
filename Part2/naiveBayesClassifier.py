import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt

from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def naive_bayes_content(encoded, labels, split):

    articles = np.array(encoded)
    labels = np.array(labels)

    X_train = articles[:split]
    y_train = labels[:split]
    X_val = articles[split:]
    y_val = labels[split:]

    y_train = y_train.astype('int')
    y_val = y_val.astype('int')

    nb = ComplementNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_val)

    accuracy = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred, zero_division=1,average='weighted')
    recall = metrics.recall_score(y_val, y_pred, zero_division=1,average='weighted')
    f1_score = metrics.f1_score(y_val, y_pred, zero_division=1, average='weighted')

    conf_matr = metrics.confusion_matrix(y_val, y_pred)
    class_report = metrics.classification_report(y_val, y_pred, zero_division=1)

    print('nb accuracy: {:0.5f}'.format(accuracy))
    print('nb precision: {:0.5f}'.format(precision))
    print('nb recall: {:0.5f}'.format(recall))
    print('nb f1-score: {:0.5f}'.format(f1_score))

    print('nb confusion matrix:\n', conf_matr,'\n')
    print('nb report:\n', class_report)


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

