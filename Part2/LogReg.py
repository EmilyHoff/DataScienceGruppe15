from distutils.ccompiler import new_compiler
from multiprocessing import reduction
import sys
import pandas as pd
from traitlets import default
import numpy as np
import nltk
import re
from collections import defaultdict
import math
import re 
import os

import matplotlib.pyplot as plt
#from matplotlib import pyplot
import random
import pickle

from sklearn.linear_model import LinearRegression

nltk.download('punkt')

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression

#from padding import padder

def linReg(df):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(df["Article encoded"])
    encoded_articles = tokenizer.texts_to_sequences(df["Article encoded"])
    X = pad_sequences(encoded_articles, padding='post')
    labels = np.where(df["Labels"]=="fake", 1, 0)
    reg = LogisticRegression().fit(X[:100], labels[:100])
    print(f"len of padded articles is {len(X)}")
    count = 0
    for x in range(139):
        print(f"Prediction: {reg.predict(X[100+x].reshape(1,-1))} Label: {labels[100+x]}")
        if reg.predict(X[100+x].reshape(1,-1))[0] == labels[100+x]:
            count +=1
        else:
            count = count-1
            
    
    print(f"Count is {count}")
    print(f"Score is: {reg.score(X[200:], labels[200:])}")