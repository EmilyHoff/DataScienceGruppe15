def classifierRelOrFake(df):
    for x in range(0, len(df)):
        if df['type'][x] == 'reliable' or df['type'][x] == 'clickbait' or df['type'][x] == 'political':
            df['type'][x] = 1 
        else:
            df['type'][x] = 0 
    return df