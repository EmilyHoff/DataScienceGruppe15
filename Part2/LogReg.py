from distutils.ccompiler import new_compiler
from multiprocessing import reduction
import sys
from unittest.util import _MAX_LENGTH
import pandas as pd
from traitlets import default
import numpy as np
import nltk
import re
from collections import defaultdict
import math
import re 
import os

#import matplotlib.pyplot as plt
#from matplotlib import pyplot
import random
import pickle

from sklearn.linear_model import LinearRegression

nltk.download('punkt')

import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences


sys.path.insert(0,"../")

def padSequences(X,maxLen):
    ret = []
    for x in X:
        if len(x) < maxLen:
            while len(x) < maxLen:
                x.append([0,0,0])
        ret.append(x)
    return ret

def logReg(df):
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
    
    print(f"length of lists = {set([len(x) for x in X])}")
    print(f"input with zeros: {X}")
    
    y = np.array(df["type"])
    
    reg = LogisticRegression(random_state=0).fit(X[:50],y[:50])
    
    print(f"len of padded articles is {len(X)}")
    count = 0
    print(f"single {len(reg.predict(X[:50]))}")
    y_pred = reg.predict(np.array(X[50:]))
        
    print(classification_report(y[50:],y_pred))