import Part1.zipfsLaw as zip
import Part1.stemming as stem
import Part1.regexFiltering as regex
import Part1.stopwords as stop

from collections import Counter
import matplotlib.pyplot as plt
from cleantext import clean
from sklearn import metrics
import seaborn as sns
import numpy as np

def countWords(df):
    '''Counts the number of unique words in the content column in a dataframe'''
    words = []
    for y in range(0, len(df)):
        words += ((df['content'][y]).split())
    return len(Counter(words))

def cleanGraph(df):
    '''compiles a bar plot illustrationg the unique words at each level of the
    implimented preprocessing pipeline, only has stopwords removal and filtering using
    clean_text library'''
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
                                replace_with_url="<URL>",
                                replace_with_number="<NUM>")
    words.append(countWords(df))

    #plot the effect of the cleaning
    plt.bar(['Raw','Stopwords', 'Filtering'], words)
    plt.xlabel("Cleaning Step")
    plt.ylabel("Unique Words")
    plt.show()
    return

def thoroughCleanGraph(df):
    '''Compiles the barplot of the unique words at each cleaning step in our
    so called thorough pipeline, with steps such as zipf's law, stemming and regexfilter'''
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
    plt.bar(['Raw','Zipfs Law', 'Stopwords', 'Filtering', 'Stemming'], words)
    plt.xlabel("Cleaning Step")
    plt.ylabel("Unique Words")
    plt.show()
    return 

def confusionMetric(labels, split, y_pred, title):
    '''Compiles the confusion metric for a models, based on it's
    predicted values and the true values from the dataset'''
    y_true = labels[split:]

    #confusion matrix - Predict by authors
    cm = metrics.confusion_matrix(y_true, y_pred, normalize='all')
    sns.heatmap(cm/np.sum(cm), annot=True)
    plt.xlabel("Predicted values")
    plt.ylabel("True values")
    plt.title(title)

    plt.show()