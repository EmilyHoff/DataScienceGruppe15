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
#from Part2 import LogReg
#from Part2 import padding
pd.options.mode.chained_assignment = None

sys.path.insert(0,"Part1/")
sys.path.insert(0,"Part2/")

def cleanChunkyDF(filename, chunkSz, nrows, sep):
    #if sep== none you're parsing a .csv file
    #if you wish to read a .tsv file sep='\t'
    if sep == None:
        if nrows == None:
            reader = pd.read_csv(filename, iterator=True, chunksize=chunkSz)
        else:
            reader = pd.read_csv(filename, iterator=True, chunksize=chunkSz, nrows=nrows)
    else:
        if nrows == None:
            reader = pd.read_csv(filename, iterator=True, chunksize=chunkSz, sep=sep)
        else:
            reader = pd.read_csv(filename, iterator=True, chunksize=chunkSz, nrows=nrows, sep=sep)

    df = pd.DataFrame()

    for chunk in reader:
        if sep == None:
            #removes duplicats and articles without labels
            chunk.drop_duplicates(subset='content', inplace=True, ignore_index=True)
            chunk = chunk[chunk['type'].apply(lambda x: isinstance(x, str))].drop(columns=['Unnamed: 0']).reset_index(drop=True)
            #Cleaning and preprocessing
            df = pd.concat([df, clean.cleaning(chunk)], ignore_index=True)
        else: #for LAIR tsv file case
            chunk.columns = ['ID', 'type', 'content', 'subjecs', 'speaker',
                            'job of speaker', 'state', 'party affiliation', 'barely true counts',
                            'false counts', 'half true counts', 'mostly true counts',
                            'pants on fire', 'context']
            #removes duplicats
            chunk.drop_duplicates(subset=['content'], inplace=True, ignore_index=True)
            #Cleaning and preprocessing
            df = pd.concat([df, clean.cleaning(chunk)], ignore_index=True)

    return df

train = cleanChunkyDF("train.csv", 1000,2000,None) #ændre chunk size og antal rækker der skal læses
test = cleanChunkyDF("train.tsv", 100,1000,"\t")

print(f"The set of labels in train: {set(train['type'])}")
print(f"The set of labels in test: {set(test['type'])}")


train = binaryLable.classifierRelOrFake(train)
test = binaryLable.classifierRelOrFake(test)

print(f"Imbalance in train {sum(train['type'].tolist())/len(train['type'].tolist())}")
print(f"This is imbalance in test {sum(test['type'].tolist())/len(test['type'].tolist())}")

df,split = binaryLable.combine(train,test)

df = df.drop(columns=[col for col in df.columns if col not in ['type', 'content']])

encoded,embedding_matrix,vocab_size = formatting.format(df)
advModels.bert(encoded,embedding_matrix,df['type'].tolist(),vocab_size,split)
