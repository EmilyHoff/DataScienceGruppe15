from distutils.ccompiler import new_compiler
from lib2to3.pgen2 import token
from multiprocessing import reduction
import sys
from tokenize import Token
from unittest.util import _MAX_LENGTH
import pandas as pd
from traitlets import default
import numpy as np
import nltk
import re
from collections import defaultdict
import math
import re
import os

from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
import gensim.downloader as api
from gensim import utils
import nltk

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import tensorflow as tf

#import matplotlib.pyplot as plt
#from matplotlib import pyplot
import random
import pickle

#import fasttext

nltk.download('punkt')


import pandas as pd
import nltk
import re
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import gensim.downloader as api
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('punkt')



def format(fullCorpus=None):
    with open("words.txt","w") as f:
        for x in fullCorpus["content"]:
            f.write(x)
            
    model = fasttext.train_unsupervised("words.txt",dim=150)
    embedding_matrix = np.zeros((len(model.words)+1, 150))

    wordMapper = defaultdict(int)
    wordMapper["0"] = 0
    embedding_matrix[0] = np.zeros(150)
    print(f"This is the embedding matrix: {embedding_matrix}")
    for count,word in enumerate(model.words):
        wordMapper[word] = count+1
        embedding_matrix[count+1] = model.get_word_vector(word)
        
    encoded = []
    for y in fullCorpus["content"]:
        article = []
        for x in y.split(" "):
            article.append(wordMapper[x])
        encoded.append(article)
    
    encoded = tf.keras.utils.pad_sequences(encoded,padding="post",truncating="post")  

    return encoded,embedding_matrix,len(model.words)


def bow(df):
    # Extract the article content and labels
    articles = df['content'].tolist()
    labels = df['type'].tolist()

    # Create a CountVectorizer object to encode the articles
    vectorizer = CountVectorizer(analyzer="char_wb")

    # Fit the vectorizer on the articles to learn the vocabulary
    #vectorizer.fit(articles)
    
    # Transform the articles into bag-of-words vectors
    encoded_articles = vectorizer.fit_transform(articles)
    encoded_articles = encoded_articles.toarray().tolist()

    return encoded_articles, labels