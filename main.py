import sys
import pandas as pd
import numpy as np
import nltk
import re
from collections import defaultdict
import sklearn.model_selection as sk
import math
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download("wordnet")
nltk.download('omw-1.4')
import sys

from Part1 import dataExploration
from Part1 import clean

from Part2 import binaryLable
from Part2 import simpleAuthors
from Part2 import naiveBayesClassifier

from Part2 import formatting
from Part2 import LogReg
from Part2 import BERT
from Part2 import binaryLable
from Part2 import SVM
#from Part2 import padding
# from Part1 import uniqueWords
pd.options.mode.chained_assignment = None

sys.path.insert(0,"Part1/")
sys.path.insert(0,"Part2/")

def cleanChunkyDF(filename, chunkSz,size):
    if size != None:
        reader = pd.read_csv(filename, iterator=True, chunksize=chunkSz)
    else:
        reader = pd.read_csv(filename, iterator=True, chunksize=chunkSz,nrows=size)
    df = pd.DataFrame()

    for chunk in reader:
        #removes duplicats and articles without labels
        chunk.drop_duplicates(subset='content', inplace=True, ignore_index=True)
        chunk = chunk[chunk['type'].apply(lambda x: isinstance(x, str))].drop(columns=['Unnamed: 0']).reset_index(drop=True)
        #Cleaning and preprocessing 
        df = pd.concat([df, clean.cleaning(chunk)])
    return df


df = cleanChunkyDF("news_sample.csv", 30,None) #ændre chunk size og antal rækker der skal læses
df = binaryLable.classifierRelOrFake(df)
BERT.bert(df)

df.to_csv("BeforeWordEmbedding.csv")
dfEncoded = formatting.format(fullCorpus=df,labels=df["type"].tolist(),loadModel=True,mappingName="newsSampleEncoded") #Lav word embedding returner som ny dataframe husk at give labels og 
    
#dfEncoded.to_csv("Encoded_articles.csv")

LogReg.logReg(dfEncoded)

#SVM.supportVectorMachine(dfEncoded)







def padSequences(X,maxLen):
    ret = []
    for x in X:
        if len(x) < maxLen:
            while len(x) < maxLen:
                x.append([0,0,0])
        ret.append(x)
    return ret