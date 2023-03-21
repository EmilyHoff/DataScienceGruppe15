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


    for y in ["content"]:
        for x in range(0,len(df)):
            try:
                if df["type"][x].lower() == "fake":
                    fakeTotal +=1
                    sentences = sent_tokenize(df[y][x])
                    words = [word_tokenize(sentence.lower()) for sentence in sentences]
                    words = words[0]
                    print(words)

                    tagged_words = [nltk.pos_tag(sentence) for sentence in words]
                    print(f"tags: {tagged_words}")
                    proper_nouns = []
                    for sentence in tagged_words:
                        for word, tag in sentence:
                            if tag == 'NNP': # NNP denotes proper noun
                                proper_nouns.append(word)
                    print(proper_nouns)
                    propNounsFake += len(set(proper_nouns))


                else:
                    elseTotal +=1
                    sentences = sent_tokenize(df[y][x])
                    words = [word_tokenize(sentence.lower()) for sentence in sentences]
                    words = words[0]
                    print(words)

                    tagged_words = [nltk.pos_tag(sentence) for sentence in words]
                    print(f"tags: {tagged_words}")
                    proper_nouns = []
                    for sentence in tagged_words:
                        for word, tag in sentence:
                            if tag == 'NNP': # NNP denotes proper noun
                                proper_nouns.append(word)
                    print(proper_nouns)
                    propNounsElse += len(set(proper_nouns))
            except:
                pass

    print(f"Prop nouns fake {propNounsFake/fakeTotal} else: {propNounsElse/elseTotal}")

def uniqueGraph(df):
    fakeArticles = []
    reliableArticles = []

    fakeWords = []
    reliableWords = []

    #for y in df["content"]:
    for x in range(0, len(df)):
        if df["type"][x].lower() == "fake":
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


    for x in range(0, len(df)):
        result = re.search(rf"\b{word}\b", str(df["content"][x]))
        try:
            if not(result == None): #the word is found
                if df["type"][x].lower() == "fake":
                    fakeWord += 1
                else:
                    reliableWord += 1
            else:
                if df["type"][x].lower() == "fake":
                    fakeNoWord += 1
                else:
                    reliableNoWord += 1
        except:
            pass
    print("fakeword: {}\n reliableword: {}\n fakenoword: {} \n reliableNoword: {}".format(fakeWord, reliableWord, fakeNoWord, reliableNoWord))
    print(df.shape)
    #percentage of fake articles with the word out of all fake articles
    preFake = (1 - (fakeNoWord/(fakeNoWord + fakeWord)))*100
    print("Percentage of fake articles with the word: {}%".format(preFake))

    #percentage of reliable articles with the word out of all reliabel articles
    preReliable = (1 - (reliableNoWord/(reliableNoWord + reliableWord)))*100
    print("Percentage of reliable articles with the word: {}%".format(preReliable))

    #out of all articles with the word X% of them are fake
    fakeWordCorrelation = (fakeWord/(fakeWord + reliableWord))*100
    print(fakeWordCorrelation)


def exclamationFunction(df):

    fakeExclamations = []
    nonFakeExclamations = []

    fakeNoExclamations = 0
    nonFakeNoExclamations = 0
    k = 0

    for x in range(0, len(df)):
        if str(df['type'][x]).lower() == 'fake':
            k += 1
        excl = re.findall('!', df['content'][x])
        if len(excl) == 0:
            if str(df['type'][x]).lower() == 'fake':
                fakeNoExclamations += 1
            else:
                nonFakeNoExclamations += 1
        else:
            if str(df['type'][x]).lower() == 'fake':
                fakeExclamations.append(len(excl))
            else:
                nonFakeExclamations.append(len(excl))


    fakeExclMean = stats.mean(fakeExclamations)
    nonFakeExclMean = stats.mean(nonFakeExclamations)

    fNE = (fakeNoExclamations / k)*100

    print("If the article is fake and has exclamation marks, there are on average {} of them".format(fakeExclMean))
    print("If the article isn't fake, and has exclamation marks, there are on average {} of them".format(nonFakeExclMean))

    print("Of the {} total articles, {} of them are fake".format(len(df), k))
    print("Of the {} fake articles, {} don't have exclamation marks in - {}%".format(k, fakeNoExclamations, fNE))

    # 155/250 articles are fake = 62% of them
    # of 155 articles, 84.5% of them have exclamation
    ifFake = ((k/len(df)) * (len(fakeExclamations)/k)) / (
                (k/len(df)) * (len(fakeExclamations)/k) + (
                (1 - (k/len(df))) * (1 - (len(fakeExclamations)/k)))) * 100

    # remaining 38% of articles, not fake
    # 18.9% of these have exclamation marks
    ifNonFake = (((len(df) - k)/len(df)) * (len(nonFakeExclamations)/(len(df) - k))) / (
                (((len(df) - k)/len(df)) * (len(nonFakeExclamations)/(len(df) - k))) + (
                (1 - ((len(df) - k)/len(df))) * (1 - (
                len(nonFakeExclamations)/(len(df) - k))))) * 100

    print("If an article is fake, there is a {}% chance of them have exclamation marks".format(ifFake))
    print("If an article isn't fake, there is a {}% chance that it has exclamation marks".format(ifNonFake))


df = pd.read_csv('news_sample.csv')
exclamationFunction(df)


