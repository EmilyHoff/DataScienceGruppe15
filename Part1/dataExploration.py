import numpy as np
import pandas as pd
import nltk
from collections import defaultdict
import math
import re 
from nltk.tokenize import word_tokenize, sent_tokenize

pd.options.mode.chained_assignment = None

nltk.download('punkt')
nltk.download('stopwords')




def exploringData(df):
    propNounsFake = 0
    fakeTotal = 0

    propNounsElse = 0
    elseTotal = 0

    
    for y in ["content"]:
        for x in range(0,len(df)):
            try:
                if df["type"][x].lower() == "fake":
                    fakeTotal +=1
                    sentences = sent_tokenize(df[y][x])
                    words = [word_tokenize(sentence.lower()) for sentence in sentences]
                    words = words[0]
                    print(words)

                    tagged_words = [nltk.pos_tag(sentence) for sentence in words]
                    print(f"tags: {tagged_words}")
                    proper_nouns = []
                    for sentence in tagged_words:
                        for word, tag in sentence:
                            if tag == 'NNP': # NNP denotes proper noun
                                proper_nouns.append(word)    
                    print(proper_nouns) 
                    propNounsFake += len(set(proper_nouns))


                else:
                    elseTotal +=1
                    sentences = sent_tokenize(df[y][x])
                    words = [word_tokenize(sentence.lower()) for sentence in sentences]
                    words = words[0]
                    print(words)

                    tagged_words = [nltk.pos_tag(sentence) for sentence in words]
                    print(f"tags: {tagged_words}")
                    proper_nouns = []
                    for sentence in tagged_words:
                        for word, tag in sentence:
                            if tag == 'NNP': # NNP denotes proper noun
                                proper_nouns.append(word)   
                    print(proper_nouns)  
                    propNounsElse += len(set(proper_nouns))
            except:
                pass

    print(f"Prop nouns fake {propNounsFake/fakeTotal} else: {propNounsElse/elseTotal}")