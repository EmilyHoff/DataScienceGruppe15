from multiprocessing.dummy import active_children
from sklearn import svm
from sklearn.metrics import classification_report
import sys
import numpy as np


def supportVectorMachine(df):
    max_len = max([len(seq) for seq in df["Article encoded"]])
    max_len = 250
    padded_list = [seq + [0.0] * (max_len - len(seq)) for seq in df["Article encoded"]]
    padded_list = [x[:max_len] for x in padded_list]
    X = []
    
    for li in padded_list:
        oneInput = []
        for x in li:
            if isinstance(x,list):
                oneInput.append(x[0])
            else:
                oneInput.append(x)
        X.append(oneInput)
    
    X = np.array(X)
    X = X.reshape((X.shape[0], -1))
    
    y = np.array(df["type"])
    
    clf = svm.SVC()
    clf.fit(X[:100],y[:100])
    print(clf.score(X[100:],y[100:]))
    
    

