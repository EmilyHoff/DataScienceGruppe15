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
from Part2 import baselineModels

from Part2 import formatting
#from Part2 import LogReg
#from Part2 import BERT
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

trainDf = pd.read_csv('train.csv')
valDf = pd.read_csv('val.csv')
#print(f"Passing this df {df}")
print(f"Length of columns: {len(trainDf['content'])} {len(trainDf['type'])}")

tDf = binaryLable.classifierRelOrFake(trainDf)
vDf = binaryLable.classifierRelOrFake(valDf)

#simpleAuthors.predictByAuthors(tDf, vDf)
#simpleAuthors.predictByMeta(df)

#Data Exploration
'''dataExploration.exploringData(df)
dataExploration.uniqueWords(df)
dataExploration.fakenessFromWord(df, "bitcoin")
exclamationDf = pd.read_csv("news_cleaned_2018_02_13.csv", nrows=10000)
exclamationDf = binaryLable.classifierRelOrFake(exclamationDf)
dataExploration.exclamationFunction(exclamationDf)'''

#dfEncoded = formatting.format(fullCorpus=df,labels=df["type"].tolist(),loadModel=True,mappingName="newsSampleEncoded") #Lav word embedding returner som ny dataframe husk at give labels og

# tDf = formatting.oneHotEncode(tDf)
# vDf = formatting.oneHotEncode(vDf)
# baselineModels.perceptronClassifier(tDf)
#df = BERT.bert(df)

#logReg.logReg(pd.read_csv("articlesEncoded.csv"))
