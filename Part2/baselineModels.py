import re
import pandas as pd
import numpy as np

from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn import metrics

def perceptronClassifier(df):

    content = df['content']
    y = df['type'].to_numpy()

    x = []
    for i in range(0, len(content)):
        x.append(content[i])
    x = np.array(x)

    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)
    percep = Perceptron()

    for i in range(0, len(y_train)):
        for j in range(0, len(y_train)):
            print(y_train[i])
            print(type(y_train[i]))
            break

    percep.fit(X_train, y_train)
    y_pred = percep.predict(X_val)
    accuracy = metrics.accuracy_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred, average='weighted', zero_division=1)

    print('acc',accuracy)
    print('f1',f1_score)
