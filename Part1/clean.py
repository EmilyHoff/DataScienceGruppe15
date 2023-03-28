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
    df = zip.zipfsFiltering(df)
    words.append(countWords(df))
    df = stop.removeStopwords(df)
    words.append(countWords(df))
    #df = regex.keywordFiltering(df)
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
    df = stem.applyStemming(df)
    words.append(countWords(df))

    #plot the effect of the cleaning
    plt.bar(['Raw','Zipfs law','Stopwords', 'Filtering', 'Stemming'], words)
    plt.xlabel("Cleaning Step")
    plt.ylabel("Unique Words")
    plt.show()
    return df


def cleaning(df):
    #Zipfs
    for y in range(0, len(df)):
        df["content"][y] = df["content"][y].lower()
        tokens = nltk.tokenize.word_tokenize(df["content"][y])
        allWordsDist = nltk.FreqDist(w.lower() for w in tokens)

        words = [[word,dict(allWordsDist.most_common())[word]] for word in dict(allWordsDist.most_common()) if word.isalpha()]
        words = sorted(words,key=lambda k: k[1],reverse=True)
        #print(words)

        wordCount = [x[1] for x in words]
        lower = int(np.percentile(wordCount,100*(0.05)))
        upper = int(np.percentile(wordCount,100*(0.95)))

        for word in words:
            if word[1] >= upper:
                df["content"][y] = df["content"][y].replace(f" {word[0]} "," ")
                words.remove(word)
            elif word[1] <= lower:
                df["content"][y] = df["content"][y].replace(f" {word[0]} "," ")
                words.remove(word)
    '''
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
    for y in range(0,len(df)):
        #has already been lower cased from zipf's law
        df['content'][y] = re.sub(r"\t"," ",str(df['content'][y])) #Remove tab
        df['content'][y] = re.sub(r"\n"," ",str(df['content'][y])) #Remove newline
        df['content'][y] = re.sub(r" {2,}"," ",str(df['content'][y])) #Remove extra white space

        df['content'][y] = re.sub(r"((http://|https://)*(www\.)*([\w\d\._-]+)(\.[\w]{2,})(\.)*?(/[\w\d#%=&/?\.+_-]+)*(\.[\w]+)*)",
                                "<URL>", str(df['content'][y]))

        df['content'][y] = re.sub(r"\d{4}[-|\/|\\]\d{2}[-|\/|\\]\d{2}\b","<DATE>",str(df['content'][y])) #Remove Date
        df['content'][y] = re.sub(r"\b\d{2}[-|\/|\\]{1}\d{2}[-|\/|\\]{1}\d{2}\b","<DATE>",str(df['content'][y]))
        df['content'][y] = re.sub(r"\b\d{2}[-|\/|\\]{1}\d{2}[-|\/|\\]{1}\d{4}\b","<DATE>",str(df['content'][y]))
        df['content'][y] = re.sub(r"((jan[uary]*|feb[ruary]*|mar[ch]*|apr[il]*|may|jun[e]*|jul[y]*|aug[ust]*|sep[tember]*|oct[ober]*|nov[ember]*|dec[ember]*) ([\d]+(\w{2})*) ?(rd|st|th+))",
                                "<DATE>", str(df['content'][y]))
        df['content'][y] = re.sub(r"\d{1,2}?(nd|rd|st|th)", "<DATE>", str(df['content'][y])) #match format num(nd, th, rd, st)
        df['content'][y] = re.sub(r"\b[\w\.\-]+[\d\w]+?[@][\w]+?[\.][a-z]{2,}\b", "<EMAIL>", str(df['content'][y])) #Remove email

        df['content'][y] = re.sub(r"(@|\(@)[^\s]+","<Twitter>",str(df['content'][y])) #Removes twitter

        df['content'][y] = re.sub(r"[0-9]+[\.|,|:|0-9]*","<NUM>",str(df['content'][y])) #Remove num

        df['content'][y] = re.sub(r"[^\s\w\d]", "", str(df['content'][y])) #remove punctuation

    '''

    #Fjerner stopword

    stop_words = set(stopwords.words('english'))
    for y in range(0, len(df)):
        tokens = word_tokenize(df["content"][y])
        df["content"][y] = ' '.join([word for word in tokens if not word in stop_words])

    #Laver stemming og lemmatizing

    for i in range(0, len(df)):
        tokens = word_tokenize(df["content"][i])
        tokenTags = nltk.tag.pos_tag(tokens)
        ret = []
        for (x,y) in tokenTags:
            if "VB" in y:
                ret.append(PorterStemmer().stem(x))
            else:
                ret.append(WordNetLemmatizer().lemmatize(x))
        df["content"][i] = ' '.join(ret)
    return df
