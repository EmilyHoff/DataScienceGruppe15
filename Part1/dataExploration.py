import numpy as np
import pandas as pd
import nltk
from collections import defaultdict
import math
import statistics as stats
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import matplotlib.pyplot as plt


pd.options.mode.chained_assignment = None

nltk.download('punkt')
nltk.download('stopwords')

def exploringData(df):
    propNounsFake = 0
    fakeTotal = 0

    propNounsElse = 0
    elseTotal = 0

    #allows us to compare
    types = df['type'].tolist()

    for x in range(0,len(df)):
        if types[x] == 0:
            fakeTotal +=1
            sentences = sent_tokenize(str(df['content'][x]))
            words = [word_tokenize(sentence.lower()) for sentence in sentences]
            words = words[0]

            tagged_words = [nltk.pos_tag(words)]
            proper_nouns = []
            for sentence in tagged_words:
                for word, tag in sentence:
                    if tag == 'NNP': # NNP denotes proper noun
                        proper_nouns.append(word)
            propNounsFake += len(set(proper_nouns))

        else:
            elseTotal +=1
            sentences = sent_tokenize(str(df['content'][x]))
            words = [word_tokenize(sentence.lower()) for sentence in sentences]
            words = words[0]

            tagged_words = [nltk.pos_tag(words)]
            proper_nouns = []
            for sentence in tagged_words:
                for word, tag in sentence:
                    if tag == 'NNP': # NNP denotes proper noun
                        proper_nouns.append(word)
            propNounsElse += len(set(proper_nouns))

    print(f"Prop nouns fake {propNounsFake/fakeTotal} else: {propNounsElse/elseTotal}")

def uniqueWords(df):
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

#this function assumes that stopwrods has been removed, data has been clean, but not stemmed
def fakenessFromWord(df, word):

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

def exclamationFunction(df):

    fakeExclamations = []
    nonFakeExclamations = []

    fakeNoExclamations = 0
    nonFakeNoExclamations = 0

    types = df['type'].tolist()
    fakeArticles = types.count(1)

    for x in range(0, len(df)):
        excl = re.findall('!', df['content'][x])
        if bool(excl) == False: #the list is empty
            if types[x] == 0:
                fakeNoExclamations += 1
            else:
                nonFakeNoExclamations += 1
        else:
            if types[x] == 0:
                fakeExclamations.append(len(excl))
            else:
                nonFakeExclamations.append(len(excl))

    fakeExclMean = stats.mean(fakeExclamations)
    nonFakeExclMean = stats.mean(nonFakeExclamations)

    fNE = (fakeNoExclamations / fakeArticles)*100

    print("If the article is fake and has exclamation marks, there are on average {} of them".format(fakeExclMean))
    print("If the article isn't fake, and has exclamation marks, there are on average {} of them".format(nonFakeExclMean))

    print("Of the {} total articles, {} of them are fake".format(len(df), fakeArticles))
    print("Of the {} fake articles, {} don't have exclamation marks in - {}%".format(fakeArticles, fakeNoExclamations, fNE))

    # 155/250 articles are fake = 62% of them
    # of 155 articles, 84.5% of them have exclamation
    #ifFake = ((fakeArticles/len(df)) * (len(fakeExclamations)/fakeArticles)) / (
    #            (fakeArticles/len(df)) * (len(fakeExclamations)/fakeArticles) + (
    #            (1 - (fakeArticles/len(df))) * (1 - (len(fakeExclamations)/fakeArticles)))) * 100
    ifFake = (fakeArticles-len(fakeExclamations))/fakeArticles
    # remaining 38% of articles, not fake
    # 18.9% of these have exclamation marks
    #ifNonFake = (((len(df) - fakeArticles)/len(df)) * (len(nonFakeExclamations)/(len(df) - fakeArticles))) / (
    #            (((len(df) - fakeArticles)/len(df)) * (len(nonFakeExclamations)/(len(df) - fakeArticles))) + (
    #            (1 - ((len(df) - fakeArticles)/len(df))) * (1 - (
    #            len(nonFakeExclamations)/(len(df) - fakeArticles))))) * 100
    ifNonFake = ((len(df) - fakeArticles) - len(nonFakeExclamations))/(len(df) - fakeArticles)

    print("If an article is fake, there is a {}% chance of them have exclamation marks".format(ifFake))
    print("If an article isn't fake, there is a {}% chance that it has exclamation marks".format(ifNonFake))


