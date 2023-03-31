# DataScienceGruppe15

We are Group 15, this is our exam project in Data Science, hand-in 31st March 2023.

We were tasked with training a simple and advanced model on a dataset of articles, classified according to their relative reliability, to then use for predicting binary labels on a different data set.

It you wish to reproduce our dataexploration and graphs, uncomment the specific parts in the main.py file.

The data sets used are the FakeNewsCorpus, the entire 27GB. Here we split the first 100.000 rows into training, test and validation sets (80%, 10%, 10%). You can acces the corpus here:

https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0.

Simply download the files, unzip them and the resulting file, news_cleaned_2018_02_13.csv, is the one our entire models are trained on. To reproduce our test scores, you'll need the LIAR dataset, you can find it here:

https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

We test on the first 10.000 articles from the train.tsv file.

If you ensure that news_cleaned_2018_02_13.csv and train.tsv is in the same directory as main.py then you compile the code, everything will run smoothly.

To run this program, navigate to the directory in where main.py lies then run "python3 main.py" fromm your terminal.

Sit back, and enjoy :) 
