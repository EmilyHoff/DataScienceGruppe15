from multiprocessing.dummy import active_children
from sklearn import svm
from sklearn.metrics import classification_report
import sys

def padSequences(X,maxLen):
    ret = []
    for x in X:
        if len(x) < maxLen:
            while len(x) < maxLen:
                x.append([0,0,0])
        ret.append(x)
    return ret


def supportVectorMachine(df):
    XTrain = df["Article encoded"][:100]
    yTrain = df["type"][:100]
    
    XTest = df["Article encoded"][130:]
    yTest = df["type"][130:]
    
    XTrain = padSequences(XTrain,max([len(x) for x in XTrain]))
    XTest = padSequences(XTest,max([len(x) for x in XTest]))
    
    for x in XTrain:
        print(x)
    
    clf = svm.SVC()
    clf.fit(XTrain,yTrain)
    print(clf.score(XTest,yTest))
    
    

