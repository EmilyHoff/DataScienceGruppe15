def classifierRelOrFake(df):
    labels = df['type'].tolist()
    for x in range(0, len(df)):
        if (labels[x] == 'reliable') or (labels[x] == 'clickbait') or (labels[x] == 'political'):
            df['type'][x] = 1
        else:
            df['type'][x] = 0
    return df
