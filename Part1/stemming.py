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

def applyStemming(df):
    tokens = word_tokenize(df["content"])
    tokenTags = nltk.tag.pos_tag(tokens)
    ret = []
    for (x,y) in tokenTags:
        if "VB" in y:
            ret.append(PorterStemmer().stem(x))
        else:
            ret.append(WordNetLemmatizer().lemmatize(x))
    df["content"] = ' '.join(ret)
    return df
