import Part1.zipfsLaw as zip
import Part1.stemming as stem
import Part1.regexFiltering as regex
import Part1.stopwords as stop
import matplotlib.pyplot as plt
from collections import Counter
from cleantext import clean
import re
import nltk
import numpy as np

#NB! does only handel one chunk at the time

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')

def countWords(df):
    words = []
    for y in range(0, len(df)):
        words += ((df['content'][y]).split())
    return len(Counter(words))

def cleaningGraph(df):
    words = []
    words.append(countWords(df))
    df = stop.removeStopwords(df)
    words.append(countWords(df))

    for i in range(0, len(df)):
        df['content'][i] = clean(df['content'][i],
                                lower=True,
                                no_emails=True,
                                no_urls=True,
                                no_numbers=True,
                                no_punct=True,
                                normalize_whitespace=True,
                                replace_with_email="<EMAIL>",
                                replace_with_punct=' ',
                                replace_with_url="<URL>",
                                replace_with_number="<NUM>")
    words.append(countWords(df))

    #plot the effect of the cleaning
    plt.bar(['Raw','Stopwords', 'Filtering'], words)
    plt.xlabel("Cleaning Step")
    plt.ylabel("Unique Words")
    plt.show()
    return df

def cleaning(df):
    #cleaning
    for i in range(0, len(df)):
        df['content'][i] = clean(df['content'][i],
                                lower=True,
                                no_emails=True,
                                no_urls=True,
                                no_numbers=True,
                                no_punct=True,
                                normalize_whitespace=True,
                                replace_with_email="<EMAIL>",
                                replace_with_url="<URL>",
                                replace_with_number="<NUM>")

    #stopwords removal
    stop_words = set(stopwords.words('english'))
    for y in range(0, len(df)):
        tokens = word_tokenize(df["content"][y])
        df["content"][y] = ' '.join([word for word in tokens if not word in stop_words])
    return df