import sys
import pandas as pd
import numpy as np
import nltk
import re
from collections import defaultdict
import math
import re 

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import nltk

nltk.download('punkt')




def generateModel(df,**kwargs):
    vocabSize = []
    for count,x in enumerate(df):
        for y in df["content"][count].split(" "):
            vocabSize.append(y)
    vocabSize = set(vocabSize)

    





    return df




