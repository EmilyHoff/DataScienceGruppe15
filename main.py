import sys
import pandas as pd
from Part1 import regexFiltering
from Part1 import zipfsLaw
from Part1 import dataExploration

sys.path.insert(0,"Part1/")

df = pd.read_csv("news_sample.csv")[:5]
df.drop_duplicates(subset='content', inplace=True,ignore_index=True)

for x in range(0,len(df)):
    df.iloc[x] = zipfsLaw.commonWords(df.iloc[x])
    df.iloc[x] = regexFiltering.basicFiltering(df.iloc[x])
    #tilføj funktioner husk kun at give en linje

#dataExploration.exploringData(df)

df.to_csv("Results.csv")