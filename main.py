import sys
import pandas as pd
import numpy as np
import nltk
import statistics as stats
import re
from collections import defaultdict
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

from Part1 import regexFiltering
from Part1 import zipfsLaw
from Part1 import dataExploration
from Part1 import stopwords
from Part1 import stemming

from Part2 import formatting
from Part2 import LogReg
#from Part2 import padding
# from Part1 import uniqueWords
pd.options.mode.chained_assignment = None


sys.path.insert(0,"Part1/")
sys.path.insert(0,"Part2/")


df = pd.read_csv("news_sample.csv")
df.drop_duplicates(subset='content', inplace=True,ignore_index=True)

# print(df)
for x in range(0,len(df)):
    df.iloc[x] = zipfsLaw.zipfsFiltering(df.iloc[x])
    df.iloc[x] = stopwords.removeStopwords(df.iloc[x])
    df.iloc[x] = regexFiltering.keywordFiltering(df.iloc[x])
    df.iloc[x] = stemming.applyStemming(df.iloc[x])
    pass

formatting.generateModel(fullCorpus=df,loadModel=True,mappingName="newsSampleEncoded").to_csv("articlesEncoded.csv")
LinearReg.linReg(pd.read_csv("articlesEncoded.csv"))

#df.to_csv("Results.csv")


