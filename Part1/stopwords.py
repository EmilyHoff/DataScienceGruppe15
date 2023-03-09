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

def removeStopwords(df):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(df["content"])
    df["content"] = ' '.join([word for word in tokens if not word in stop_words])
    return df