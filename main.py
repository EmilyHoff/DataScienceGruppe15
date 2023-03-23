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
from Part2 import BERT
from Part2 import binaryLable
#from Part2 import padding
# from Part1 import uniqueWords
pd.options.mode.chained_assignment = None

sys.path.insert(0,"Part1/")
sys.path.insert(0,"Part2/")

def cleanChunkyDF(filename, chunkSz, iterations):
    reader = pd.read_csv(filename, iterator=True, chunksize=chunkSz)
    df = pd.DataFrame()

    while(iterations > 0):
        for chunk in reader:
            #removes duplicats and articles without labels
            chunk.drop_duplicates(subset='content', inplace=True, ignore_index=True)
            chunk = chunk[chunk['type'].apply(lambda x: isinstance(x, str))].drop(columns=['Unnamed: 0']).reset_index(drop=True)
            #Cleaning and preprocessing 
            df = pd.concat([df, clean.cleaning(chunk)])
            iterations -= 1
            break
    return df

#df = cleanChunkyDF("news_cleaned_2018_02_13.csv", 10, 1)
df = cleanChunkyDF("news_sample.csv", 10, 1)

df = binaryLable.classifierRelOrFake(df)
simpleAuthors.authorNContent(df)

#formatting.format(fullCorpus=df,loadModel=True,mappingName="newsSampleEncoded").to_csv("articlesEncoded.csv")
#logReg.logReg(pd.read_csv("articlesEncoded.csv"))

'''
#prepare data for models 
df = binaryLable.classifierRelOrFake(df)

simpleAuthors.predictByAuthors(df)
naiveBayesClassifier.naive_bayes_authors(df)
# naiveBayesClassifier.naive_bayes(df, 'content')

x_test, x_val, y_test, y_val = sk.train_test_split(df['content'], df["type"], test_size=0.2, random_state=0)

#when working with the large dataset, maybe convert to csv now to avoid recompiling

x_val, x_train, y_val, y_train = sk.train_test_split(x_val, y_val, test_size=0.5, random_state=0)

#print(df)
df.to_csv("Results.csv")

'''