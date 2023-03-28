def classifierRelOrFake(df):
    #prepare data
    labels = df['type'].astype('string').tolist()
    trueLabels = ['reliable', 'clickbait', 'political', 'true', 'mostly true']
    for x in range(0, len(df)):
        if (labels[x] in trueLabels) :
            df['type'][x] = 1
        else:
            df['type'][x] = 0
    return df
