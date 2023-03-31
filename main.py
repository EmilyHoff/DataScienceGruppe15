import sys
import pandas as pd
import nltk
from collections import defaultdict
import sklearn.model_selection as sk
import sys
import numpy as np
from sklearn import metrics
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download("wordnet")
nltk.download('omw-1.4')

from Part1 import dataExploration
from Part1 import clean

from Part2 import binaryLable
from Part2 import naiveBayesClassifier
from Part2 import simpleModels
from Part2 import advModels

from Part2 import formatting

import graphs

pd.options.mode.chained_assignment = None

sys.path.insert(0,"Part1/")
sys.path.insert(0,"Part2/")

#dataExploration
'''explorDf = clean.cleanChunkyDF("news_cleaned_2018_02_13.csv", 1000, 10000, None)
trainEx, valEx = sk.train_test_split(explorDf, test_size=0.2, random_state=0)
encodedEx, splitsEx = binaryLable.combine(trainEx, valEx)

explorDf = binaryLable.classifierRelOrFake(explorDf)
dataExploration.averageAuthors(explorDf)
dataExploration.uniqueWords(explorDf)
dataExploration.fakenessFromWord(explorDf, "bitcoin")

dataExploration.plot_words(explorDf, splitsEx, model='nb')
dataExploration.plot_words(explorDf, splitsEx, model='percep')
dataExploration.plot_words(explorDf, splitsEx, model='logreg')'''

#cleaning graphs
'''graphs.cleanGraph(pd.read_csv("news_cleaned_2018_02_13.csv", nrows=10000))
graphs.thoroughCleanGraph(pd.read_csv("news_cleaned_2018_02_13.csv", nrows=10000))'''

#load fakenews corpus
df = clean.cleanChunkyDF("news_cleaned_2018_02_13.csv", 1000, 100000, None)
df = binaryLable.classifierRelOrFake(df)

#split data
train, val = sk.train_test_split(df, test_size=0.2, random_state=47)
val, test = sk.train_test_split(val, test_size=0.5, random_state=47)

#prepare data for training 
trainVal, split = binaryLable.combine(train, val)
trainTest,testSplit = binaryLable.combine(train,test)

encoded, labels = formatting.bow(trainVal)
trainTestEncoded,trainTestLabels = formatting.bow(trainTest)

#prepare the liar data for testing
liar = clean.cleanChunkyDF("train.tsv", 1000, 10000, "\t")
liar = binaryLable.classifierRelOrFake(liar)

corpusLiar,split = binaryLable.combine(train,liar)
encoded,labels = formatting.bow(corpusLiar)

#Test the models on Liar data
y_predLog, logMod = simpleModels.bow_logreg(encoded, labels, split)
advModels.LSTM(encoded,labels,split=split)
naiveBayesClassifier.naive_bayes_content(encoded,labels,split)

#Train on test data from FakeNewsCorpus
_predA, authorMod = simpleModels.predictByAuthors(trainTest,trainTestLabels,testSplit)
y_predLog, logMod = simpleModels.bow_logreg(trainTestEncoded,trainTestLabels,testSplit)
advModels.LSTM(trainTestEncoded,trainTestLabels,split=testSplit)
y_predNB, nb = naiveBayesClassifier.naive_bayes_content(trainTestEncoded,trainTestLabels,testSplit)