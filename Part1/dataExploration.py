import pandas as pd
import nltk
import math
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression


import Part2.simpleModels as simMod


pd.options.mode.chained_assignment = None

nltk.download('punkt')
nltk.download('stopwords')


def averageAuthors(df):
    '''Function for data exploration, to calculate the average number of authors
    in fake and reliable articles respectivly'''
    #prepare data
    labels = df['type'].tolist()
    df = simMod.numberOfAuthors(df)
    fakeAuthors = []
    reliableAuthors = []

    for i in range(0, len(df)):
        if labels[i] == 1: #true article
            reliableAuthors.append(df['authors'][i])
        else:
            fakeAuthors.append(df['authors'][i])

    print("Average number of authors in fake articles: ", sum(fakeAuthors)/len(fakeAuthors))
    print("Avergae number at authors in reliable article: ", sum(reliableAuthors)/len(reliableAuthors))

def uniqueWords(df):
    '''Calculates the average amount of unique words in fake and realiable articles'''
    fakeArticles = []
    reliableArticles = []

    fakeWords = []
    reliableWords = []

    types = df['type'].tolist()

    for x in range(0, len(df)):
        if types[x] == 0:
            fakeWords = Counter(sorted(word_tokenize(df["content"][x])))
            fakeArticles.append(len(fakeWords))
        else:
            reliableWords = Counter(sorted(word_tokenize(df["content"][x])))
            reliableArticles.append(len(reliableWords))

    AvFake = sum(fakeArticles)/len(fakeArticles)
    AvReliable = sum(reliableArticles)/len(reliableArticles)
    dif = (AvFake-AvReliable)/AvFake*100

    print("Unique words in fake articles: " + str(AvFake))
    print("Unique words in reliable articles: " + str(AvReliable))
    print("Difference: {} %".format(math.floor(dif)))

def fakenessFromWord(df, word):
    '''Calculatecs the correclation between a word appearing in a fake articel'''
    fakeWord = 0
    reliableWord = 0

    fakeNoWord = 0
    reliableNoWord = 0

    word = word.lower()
    types = df['type'].tolist()

    for x in range(0, len(df)):
        result = re.search(rf"\b{word}\b", str(df["content"][x]))
        try:
            if not(result == None): #the word is found
                if types[x] == 0:
                    fakeWord += 1
                else:
                    reliableWord += 1
            else:
                if types[x] == 0:
                    fakeNoWord += 1
                else:
                    reliableNoWord += 1
        except:
            pass
    print("fakeWord: {}\n reliableWord: {}\n fakeNoWord: {} \n reliableNoWord: {}".format(fakeWord, reliableWord, fakeNoWord, reliableNoWord))

    #percentage of fake articles with the word out of all fake articles
    preFake = (1 - (fakeNoWord/(fakeNoWord + fakeWord)))*100
    print("Percentage of fake articles with the word: {}%".format(preFake))

    #percentage of reliable articles with the word out of all reliabel articles
    preReliable = (1 - (reliableNoWord/(reliableNoWord + reliableWord)))*100
    print("Percentage of reliable articles with the word: {}%".format(preReliable))

    #out of all articles with the word X% of them are fake
    fakeWordCorrelation = (fakeWord/(fakeWord + reliableWord))*100
    print("Precentage of fake articles from alle articles with the word: ", fakeWordCorrelation)
    return 

def plot_words(dfNotEncoded, split, model):
    '''Plots the 5 most common words in articles labled as fake and relieable respectively'''
    articles = np.array(dfNotEncoded['content'])[:split]
    labels = np.array(dfNotEncoded['type'])[:split]
    labels = labels.astype('int')

    tfidf = TfidfVectorizer(stop_words='english')
    articles = tfidf.fit_transform(articles)

    if model == 'nb':
        nb = ComplementNB()
        nb.fit(articles, labels)

        # Get feature weights and sort them
        weights = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
        indices = np.argsort(weights)

        topbottom5_words = np.concatenate((indices[:5], indices[-5:]))
        topbottom5_weights = np.concatenate((weights[indices[:5]], weights[indices[-5:]]))

        # colours for the chart
        colours = np.array(['blue']*5+['red']*5)

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(np.arange(len(topbottom5_words)), topbottom5_weights, color=colours)

        # Set axis labels and title
        ax.set_yticks(np.arange(len(topbottom5_words)))
        ax.set_yticklabels(np.array(tfidf.get_feature_names_out())[topbottom5_words])
        ax.set_xlabel('Influence Level')
        ax.set_title('Words affecting Naive Bayes Classifier with TF-IDF encoding')

        # Display the chart
        plt.show()

    if model == 'percep':
        percep = Perceptron()
        percep.fit(articles, labels)

        weights = percep.coef_[0]
        # weights = percep.coef_[0] * np.asarray(articles.mean(axis=0)).ravel()
        indices = np.argsort(weights)

        topbottom5_words = np.concatenate((indices[:5], indices[-5:]))
        topbottom5_weights = np.concatenate((weights[indices[:5]], weights[indices[-5:]]))

        # colours for the chart
        colours = np.array(['black']*5+['green']*5)
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(np.arange(len(topbottom5_words)),topbottom5_weights, color=colours)

        # Set axis labels and title
        ax.set_yticks(np.arange(len(topbottom5_words)))
        ax.set_yticklabels(np.array(tfidf.get_feature_names_out())[topbottom5_words])
        ax.set_xlabel('Influence Level')
        ax.set_title('Words affecting Perceptron Classifier with TF-IDF encoding')

        # Display the chart
        plt.show()

    if model == 'logreg':
        logreg = LogisticRegression()
        logreg.fit(articles, labels)

        weights = logreg.coef_[0]
        indices = np.argsort(weights)

        topbottom5_words = np.concatenate((indices[:5], indices[-5:]))
        topbottom5_weights = np.concatenate((weights[indices[:5]], weights[indices[-5:]]))

        # colours for the chart
        colours = np.array(['purple']*5+['skyblue']*5)
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(np.arange(len(topbottom5_words)),topbottom5_weights, color=colours)

        # Set axis labels and title
        ax.set_yticks(np.arange(len(topbottom5_words)))
        ax.set_yticklabels(np.array(tfidf.get_feature_names_out())[topbottom5_words])
        ax.set_xlabel('Influence Level')
        ax.set_title('Words affecting Logistic Regression with TF-IDF encoding')

        # Display the chart
        plt.show()