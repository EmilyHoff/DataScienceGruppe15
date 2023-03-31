import pandas as pd

def classifierRelOrFake(df):
    '''Converts the types of the articles into binary labels'''
    labels = df['type'].astype('string').tolist()
    trueLabels = ['reliable', 'true', 'mostly-true']
    for x in range(0, len(df)):
        if (labels[x] in trueLabels) :
            df['type'][x] = 1
        else:
            df['type'][x] = 0
    return df

def combine(first, second):
    '''Combines two DataFrames'''
    combined = pd.concat([first,second],ignore_index=True)
    return combined, combined.index[first.shape[0]]
