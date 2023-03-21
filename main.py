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

from Part1 import regexFiltering
from Part1 import zipfsLaw
from Part1 import dataExploration
from Part1 import stopwords
from Part1 import stemming


from Part2 import binaryLable
from Part2 import simpleAuthors
from Part2 import naiveBayesClassifier

pd.options.mode.chained_assignment = None

sys.path.insert(0,"Part1/")

df = pd.read_csv("news_sample.csv")#[:10]

#The large dataset
#df = pd.read_csv("news_cleaned_2018_02_13.csv")

df.drop_duplicates(subset='content', inplace=True,ignore_index=True)

#print(df)

for x in range(0,len(df)):
    df.iloc[x] = zipfsLaw.zipfsFiltering(df.iloc[x])
    df.iloc[x] = stopwords.removeStopwords(df.iloc[x])
    df.iloc[x] = regexFiltering.keywordFiltering(df.iloc[x])
    df.iloc[x] = stemming.applyStemming(df.iloc[x])

    #tilføj funktioner husk kun at give en linje
#dataExploration.fakenessFromWord(df, "bitcoin")
#dataExploration.uniqueGraph(df)
#dataExploration.exploringData(df)

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