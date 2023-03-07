import sys
import pandas as pd
from Part1 import regexFiltering
from Part1 import zipfsLaw

sys.path.insert(0,"Part1/")

df = pd.read_csv("news_sample.csv")
df = regexFiltering.basicFiltering(df)
df = zipfsLaw.commonWords(df)
df.to_csv("Results.csv")