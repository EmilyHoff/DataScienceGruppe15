import re
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve
import sklearn.model_selection as sk
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


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

    #ROC-curves
    fpr, tpr, threshold = metrics.roc_curve(y_val,  y_pred)
    auc = metrics.roc_auc_score(y_val, y_pred)
    plt.plot(fpr,tpr,label="auc= " + str(auc))
    plt.legend(loc=4)
    plt.show()
    return

