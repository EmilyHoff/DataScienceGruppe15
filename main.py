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
#from Part2 import advModels

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

#cleaning the data and relabeling
df = clean.cleanChunkyDF("news_cleaned_2018_02_13.csv", 1000, 100000, None)
df = binaryLable.classifierRelOrFake(df)

#split data
train, val = sk.train_test_split(df, test_size=0.2, random_state=47)
val, test = sk.train_test_split(df, test_size=0.1, random_state=47)

#prepare data for training 
trainVal, split = binaryLable.combine(train, val)
encoded, labels = formatting.bow(trainVal)

#train models and make predictions on validation data
y_predA, authorMod = simpleModels.predictByAuthors(trainVal, labels, split)
y_predLog, logMod = simpleModels.bow_logreg(encoded, labels, split)
y_predPer, perMod = simpleModels.bow_perceptron(encoded, labels, split)
y_predNB, nb = naiveBayesClassifier.naive_bayes_content(encoded, labels, split)

#produce the ROC-curves
graphs.ROCcurve(y_predA, y_predLog, y_predPer, y_predNB, labels, split)

#produce the confusionmetric 
#graphs.confusionMetric(labels, split, y_predA, "Confusion Matrix - Predict by authors")

#Evaluation part - test on LAIR and test data
lairDf = clean.cleanChunkyDF('test.tsv', 100, 1000)
lairDf = binaryLable.classifierRelOrFake(lairDf)

encoded, labels = formatting.bow(test)
labels = np.array(labels)
encoded = np.array(encoded) 

#predictions on test dataset from FakeNewsCorpus
x_testA = simpleModels.numberOfAuthors(test)
x_testA = np.array(x_testA)
y_predA = authorMod.predict(x_testA)

