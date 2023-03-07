from enum import EnumMeta
from os import remove
import numpy as np
import pandas as pd
import nltk
from collections import defaultdict
import math
import re 
from scipy.stats import norm
pd.options.mode.chained_assignment = None

nltk.download('punkt')
nltk.download('stopwords')

def commonWords(df,quantiles=[0.05,0.95],generateGraph=True):
  df = df[:5]
  for y in ["content"]:
    for x in range(0,len(df)):
      tokens = nltk.tokenize.word_tokenize(df[y][x])
      allWordsDist = nltk.FreqDist(w.lower() for w in tokens)

      words = [[word,dict(allWordsDist.most_common())[word]] for word in dict(allWordsDist.most_common()) if word.isalpha()]
      words = sorted(words,key=lambda k: k[1],reverse=True)
      print(words)

      wordCount = [x[1] for x in words]
      lower = int(np.percentile(wordCount,100*(quantiles[0])))
      upper = int(np.percentile(wordCount,100*(quantiles[1])))

      mostCommon,leastCommon = [],[]
      
      for word in words:
        if word[1] >= upper:
          mostCommon.append(word[0])
        elif word[1] <= lower:
          leastCommon.append(word[0])
      #print(mostCommon)
      #print(leastCommon)

      for removeWord in mostCommon:

        #print(f"before = {len(df[y][x])}")
        df[y][x] = df[y][x].replace(f" {removeWord} "," ")
        #print(f"after = {len(df[y][x])}")

      for removeWord in leastCommon:
        #print(f"before = {len(df[y][x])}")
        df[y][x] = df[y][x].replace(f" {removeWord} "," ")
        #print(f"after = {len(df[y][x])}")
      
  return df


              
###     
              
              
              
              
