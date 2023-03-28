from distutils.ccompiler import new_compiler
from multiprocessing import reduction
import sys
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

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
import gensim.downloader as api
from gensim import utils
import nltk
import Part2.binaryLable as pbl

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#import matplotlib.pyplot as plt
#from matplotlib import pyplot
import random
import pickle

nltk.download('punkt')


import pandas as pd
import nltk
import re
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder
from gensim.utils import simple_preprocess
import gensim.downloader as api
import pickle

nltk.download('punkt')

def format(fullCorpus=None,labels=None,loadModel=False,mappingName=None):
    corpusVocab = None
    fullCorpusEncoded = []
    labels = fullCorpus["type"]
    if not loadModel:
        model = api.load("word2vec-google-news-300")
        fullCorpus = fullCorpus.content.apply(lambda x: simple_preprocess(str(x)))
        modelVocab = {word.lower():model[word] for word in model.key_to_index.keys()}
        corpusWords = list(set([x for li in fullCorpus for x in li]))

        corpusVocab = defaultdict(list)
        for x in corpusWords:
            try:
                corpusVocab[x] = modelVocab[x]
            except:
                pass

        corpusWords = [x for x in corpusVocab.values()]
        dimReducModel = TSNE(n_components=1,n_jobs=-1).fit_transform(np.array(corpusWords))
        corpusVocab = {list(corpusVocab.keys())[count]:list(vec) for count,vec in enumerate(dimReducModel)}

        if mappingName != None:
            with open(f"Part2/{mappingName}.pickle","wb") as f:
                pickle.dump(corpusVocab,f)
        else:
            with open("Part2/encoder.pickle","wb") as f:
                pickle.dump(corpusVocab,f)
    else:
        with open(f"Part2/{mappingName}.pickle", "rb") as f:
            corpusVocab = pickle.load(f)
        fullCorpus = fullCorpus.content.apply(lambda x: simple_preprocess(str(x)))

    for count,li in enumerate(fullCorpus):
        article = []
        print(f"current word {li}")
        for x in li:
            try:
                article.append(corpusVocab[x])
            except:
                print(f"passing with word {x}")
                pass
        #if len(article) == 0:
            #print(f"************************Count: {count} The following article has zero words appended {li}************************")
        fullCorpusEncoded.append(article)
    return pd.DataFrame(list(zip(fullCorpusEncoded,labels)),columns=["Article encoded","type"])


def oneHotEncode(df):

    content = df['content']
    labels = df['type']
    article = set()

    for words in content:
        article.update(words.split())
    vocab = sorted(article)

    # create a dictionary that maps words to their indices in the vocabulary
    word_to_index = {word: i for i, word in enumerate(vocab)}

    one_hot_vectors = []
    for words in content:
        vector = np.zeros(len(vocab))
        for word in words.split():
            vector[word_to_index[word]] = 1
        one_hot_vectors.append(vector)

    enc = OneHotEncoder(dtype=int)

    transformed = enc.fit_transform(one_hot_vectors)
    encoded = np.array(transformed.toarray())

    # print(len(encoded))
    # words = []
    # for i in range(0, len(encoded)):
    #     df['content'][i] = encoded[i]

    df = pd.DataFrame(zip(encoded, labels), columns=['content', 'type'])
    df.to_csv('one_hot_encoded.csv')
    return df

    # vi sigter efter at returnere et dataframe med 2 kolonner, content og type





