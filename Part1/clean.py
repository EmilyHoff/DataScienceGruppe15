import nltk
from cleantext import clean
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('stopwords')

def cleaning(df):
    '''Implimentatian of the pre processing pipeline; 
    filtering and removal of stopwords'''
    #filtering
    for i in range(0, len(df)):
        df['content'][i] = clean(df['content'][i],
                                lower=True, 
                                no_emails=True,
                                no_urls=True,
                                no_numbers=True,
                                no_punct=True,
                                no_emoji=True,
                                normalize_whitespace=True,
                                replace_with_email="<EMAIL>",
                                replace_with_url="<URL>",
                                replace_with_number="<NUM>")
    
    #stopwords removal
    stop_words = set(stopwords.words('english'))
    for y in range(0, len(df)):
        tokens = word_tokenize(df["content"][y])
        df["content"][y] = ' '.join([word for word in tokens if not word in stop_words])
    return df
    
def cleanChunkyDF(filename, chunkSz, nrows, sep):
    '''Given a .csv of .tsv file, a chunk size and a number of rows to read
    the function will read the file compile and clean the content 
    in chunks into a dataframe. To read a .tsv file set sep='\t' else None
    None is also a accepted input as nrows, then the entire file will be read'''
    if sep == None:
        if nrows == None:
            reader = pd.read_csv(filename, iterator=True, chunksize=chunkSz)
        else:
            reader = pd.read_csv(filename, iterator=True, chunksize=chunkSz, nrows=nrows)
    else:
        if nrows == None:
            reader = pd.read_csv(filename, iterator=True, chunksize=chunkSz, sep=sep)
        else:
            reader = pd.read_csv(filename, iterator=True, chunksize=chunkSz, nrows=nrows, sep=sep)

    df = pd.DataFrame()

    for chunk in reader:
        if sep == None:
            #removes duplicats and articles without labels
            chunk.drop_duplicates(subset='content', inplace=True, ignore_index=True)
            chunk = chunk[chunk['type'].apply(lambda x: isinstance(x, str))].drop(columns=['Unnamed: 0']).reset_index(drop=True)
            #Cleaning and preprocessing
            df = pd.concat([df, cleaning(chunk)], ignore_index=True)
        else: #for LAIR tsv file case
            chunk.columns = ['ID', 'type', 'content', 'subjecs', 'speaker',
                            'job of speaker', 'state', 'party affiliation', 'barely true counts',
                            'false counts', 'half true counts', 'mostly true counts',
                            'pants on fire', 'context']
            #removes duplicats
            chunk.drop_duplicates(subset=['content'], inplace=True, ignore_index=True)
            #Cleaning and preprocessing
            df = pd.concat([df, cleaning(chunk)], ignore_index=True)

    return df
    