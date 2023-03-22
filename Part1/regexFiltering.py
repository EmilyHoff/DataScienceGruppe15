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

def keywordFiltering(df):
    for y in range(0, len(df)):
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
        df['content'][y] = re.sub(r"\d{1,2}?(rd|st|th)", "<DATE>", str(df['content'][y])) #match format num(th, rd, st)
        df['content'][y] = re.sub(r"\b[\w\.\-]+[\d\w]+?[@][\w]+?[\.][a-z]{2,}\b", "<EMAIL>", str(df['content'][y])) #Remove email 
            
        df['content'][y] = re.sub(r".+@.+","<Twitter>",str(df['content'][y])) #Removes twitter 
            
        df['content'][y] = re.sub(r"[0-9]+[\.|,|:|0-9]*","<NUM>",str(df['content'][y])) #Remove num

        df['content'][y] = re.sub(r"[^\s\w\d]", "", str(df['content'][y])) #remove punctuation

    return df