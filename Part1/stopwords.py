import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import re

nltk.download('punkt')
nltk.download('stopwords')

def remove_stopwords(csvfile):
    stop_words = set(stopwords.words('english'))
    df = pd.read_csv(csvfile)
    contents = df['content']
    i = 0

    for cell in contents:
        tokens = word_tokenize(cell)
        filtered = [word for word in tokens if not word in stop_words]
        contents[i] = (' ').join(filtered)
        i += 1
    return df