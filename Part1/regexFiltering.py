import numpy as np
import pandas as pd
import nltk
from collections import defaultdict
import math
import re 
pd.options.mode.chained_assignment = None

nltk.download('punkt')
nltk.download('stopwords')


def basicFiltering(df):
    for y in ["content"]:
        for x in range(0,len(df)):
            try:
                df[y][x] = str(df[y][x]).lower() #make lowercase
                df[y][x] = re.sub(r" {2,}"," ",str(df[y][x])) #Remove extra white space
                df[y][x] = re.sub(r"\t"," ",str(df[y][x])) #Remove tab
                df[y][x] = re.sub(r"\n"," ",str(df[y][x])) #Remove newline
                
                #df[y][x] = re.sub(r"\bhttp.*[^ ]","<URL>",str(df[y][x])) #Remove Url
                #df[y][x] = re.sub(r"www\..+?","<URL> ",str(df[y][x]))
                #df[y][x] = re.sub(r"\b.*\.com.*\b","<URL>",str(df[y][x]))  
                df[y][x] = re.sub(r"((http://|https://)*(www\.)*([\w\d\._-]+)(\.[\w]{2,})(\.)*?(/[\w\d#%=&/?\.+_-]+)*(\.[\w]+)*)",
                                  "<URL>", str(df[y][x]))              
                
                df[y][x] = re.sub(r"\d{4}[-|\/|\\]\d{2}[-|\/|\\]\d{2}\b","<DATE>",str(df[y][x])) #Remove Date
                df[y][x] = re.sub(r"\b\d{2}[-|\/|\\]{1}\d{2}[-|\/|\\]{1}\d{2}\b","<DATE>",str(df[y][x])) 
                df[y][x] = re.sub(r"\b\d{2}[-|\/|\\]{1}\d{2}[-|\/|\\]{1}\d{4}\b","<DATE>",str(df[y][x]))
                df[y][x] = re.sub(r"((jan[uary]*|feb[ruary]*|mar[ch]*|apr[il]*|may|jun[e]*|jul[y]*|aug[ust]*|sep[tember]*|oct[ober]*|nov[ember]*|dec[ember]*) ([\d]+(\w{2})*) ?([\d]+)*)|(([\d][\d][\d][\d])-([\d][\d])-([\d][\d]))",
                                  "<DATE", str(df[y][x]))
                  
                df[y][x] = re.sub(r"\b[\w\.\-]+[\d\w]+?[@][\w]+?[\.][a-z]{2,}\b", "<EMAIL>", str(df[y][x])) #Remove email 
                
                #df[y][x] = re.sub(r".+@.+","<Twitter>",str(df[y][x])) #Can remove twitter 
                
                df[y][x] = re.sub(r"[0-9]+[\.|,|:|0-9]*","<NUM>",str(df[y][x])) #Remove num

                df[y][x] = re.sub(r"[^\s\w\d]", "", str(df[y][x])) #remove punctuation

            except:
                if not isinstance(df["content"][x],str):
                    df[y][x] = ""
    return df
