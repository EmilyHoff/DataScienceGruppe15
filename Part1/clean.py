import Part1.zipfsLaw as zip
import Part1.stemming as stem
import Part1.regexFiltering as regex
import Part1.stopwords as stop
import matplotlib.pyplot as plt
from collections import Counter

#NB! does only handel one chunk at the time 

def countWords(df):
    words = []
    for y in range(0, len(df)):
        words += ((df['content'][y]).split())
    return len(Counter(words))

def cleaningGraf(df):
    words = []
    words.append(countWords(df))
    df = zip.zipfsFiltering(df)
    words.append(countWords(df))
    df = stop.removeStopwords(df)
    words.append(countWords(df))
    df = regex.keywordFiltering(df)
    words.append(countWords(df))
    df = stem.applyStemming(df)
    words.append(countWords(df))

    #plot the effect of the cleaning
    plt.bar(['Raw', 'Zip', 'Stopwords', 'Filtering', 'Stemming'], words)
    plt.xlabel("Cleaning Step")
    plt.ylabel("Unique Words")
    plt.show()
    return df

def cleaning(df):
    df = zip.zipfsFiltering(df)
    df = stop.removeStopwords(df)
    df = regex.keywordFiltering(df)
    df = stem.applyStemming(df)
    return df