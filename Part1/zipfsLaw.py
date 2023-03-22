import sys
import pandas as pd
import numpy as np
import nltk
import re
from collections import defaultdict
import math
import re 

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download("wordnet")
nltk.download('omw-1.4')

def zipfsFiltering(df,quantiles=[0.05,0.95],generateGraph=True):
  for y in range(0, len(df)):
    df["content"][y] = df["content"][y].lower()
    tokens = nltk.tokenize.word_tokenize(df["content"][y])
    allWordsDist = nltk.FreqDist(w.lower() for w in tokens)

    words = [[word,dict(allWordsDist.most_common())[word]] for word in dict(allWordsDist.most_common()) if word.isalpha()]
    words = sorted(words,key=lambda k: k[1],reverse=True)
    #print(words)

    wordCount = [x[1] for x in words]
    lower = int(np.percentile(wordCount,100*(quantiles[0])))
    upper = int(np.percentile(wordCount,100*(quantiles[1])))
    
    for word in words:
      if word[1] >= upper:
        df["content"][y] = df["content"][y].replace(f" {word[0]} "," ")
        words.remove(word)
      elif word[1] <= lower:
        df["content"][y] = df["content"][y].replace(f" {word[0]} "," ")
        words.remove(word)
    
      
    return df
