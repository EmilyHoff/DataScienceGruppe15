import sys
import pandas as pd
import numpy as np
import nltk
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

pd.options.mode.chained_assignment = None


sys.path.insert(0,"Part1/")

df = pd.read_csv("news_sample.csv")[:10]
df.drop_duplicates(subset='content', inplace=True,ignore_index=True)
print(df)
for x in range(0,len(df)):
    df.iloc[x] = zipfsLaw.zipfsFiltering(df.iloc[x])
    df.iloc[x] = stopwords.removeStopwords(df.iloc[x])
    df.iloc[x] = regexFiltering.keywordFiltering(df.iloc[x])
    df.iloc[x] = stemming.applyStemming(df.iloc[x])
    
    #tilføj funktioner husk kun at give en linje

#dataExploration.exploringData(df)
print(df)
df.to_csv("Results.csv")