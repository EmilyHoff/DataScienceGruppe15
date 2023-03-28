import re
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn.model_selection as sk
import numpy as np
import pandas as pd
from collections import Counter


def numberOfAuthors(df):
    authors = df['authors'].tolist()
    for x in range(0, len(df)):
        if type(authors[x]) == float:
            df['authors'][x] = 0
        else:
            result = re.findall(r",", str(authors[x]))
            df['authors'][x] = len(result) + 1
    return df

def predictByAuthors(df):
    #prepare data
    df = numberOfAuthors(df)

    x = df['authors'].to_numpy()
    y = df['type'].to_numpy()

    x_train, x_val, y_train, y_val = sk.train_test_split(x, y, test_size=0.2, random_state=0)

    authorMod = LogisticRegression()

    #reshaping x_data
    x_train = np.reshape(x_train, (-1, 1))
    x_val = np.reshape(x_val, (-1, 1))

    #ensuring correct type within y_data
    y_train = y_train.astype('int')
    y_val = y_val.astype('int')

    authorMod = authorMod.fit(x_train, y_train)

    y_pred = authorMod.predict(x_val)
    y_val = y_val.astype('int')

    print("Accuracy: ", metrics.accuracy_score(y_val, y_pred))
    print("Recall: ", metrics.recall_score(y_val, y_pred, average='weighted', zero_division=1))
    print("Precision: ", metrics.recall_score(y_val, y_pred, average='weighted', zero_division=1))
    print("F1-score: ", metrics.f1_score(y_val, y_pred, average='weighted', zero_division=1))

    return


def domianReliability(df):
    fakeDomains = []
    relDomains = []

    #compile lists of domians indicating fakeness or reliability
    types = df['type'].tolist()
    domains = df['domain'].tolist()

    for x in range(0, len(df)):
        if types[x] == 0: #fake news
            fakeDomains.append(domains[x])
        else:
            relDomains.append(domains[x])

    fakeDomains = Counter(sorted(fakeDomains))
    relDomains = Counter(sorted(relDomains))

    #crossreference domains
    domainScore = []

    for domain in domains:
        if not(fakeDomains[domain] == 0): #the domain has published fake news
            if relDomains[domain] == 0: #the domain has only published fake news
                domainScore.append(0)
            else:
                score = fakeDomains[domain]/(relDomains[domain] + fakeDomains[domain])
                domainScore.append(score)
        else: # the domain has solely published rel news
            domainScore.append(1)

    domainScore = pd.DataFrame(domainScore, columns=['domainScore'])
    return pd.DataFrame(domainScore, columns=['domainScore'])


def predictByMeta(df):
    '''A baseline model for prediction of an articles authencity base on
    the domain which published it and the number of authors'''

    #prepare data
    authorsDF = numberOfAuthors(df)['authors'].to_frame()
    authorsDF = authorsDF.reset_index()

    domDf = domianReliability(df)
    domDf = domDf.reset_index()

    x = pd.concat([domDf, authorsDF], axis=1, join='inner')
    y = df['type'].to_numpy()

    x_train, x_val, y_train, y_val = sk.train_test_split(x, y, test_size=0.2, random_state=0)

    metaMod = LogisticRegression()

    #ensuring correct type within y_data
    y_train = y_train.astype('int')
    y_val = y_val.astype('int')

    metaMod = metaMod.fit(x_train, y_train)

    y_pred = metaMod.predict(x_val)
    y_val = y_val.astype('int')

    print("Accuracy: ", metrics.accuracy_score(y_val, y_pred))
    print("Recall: ", metrics.recall_score(y_val, y_pred, average='weighted', zero_division=1))
    print("Precision: ", metrics.recall_score(y_val, y_pred, average='weighted', zero_division=1))
    print("F1-score: ", metrics.f1_score(y_val, y_pred, average='weighted', zero_division=1))
    return
