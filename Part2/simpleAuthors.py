import re
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn.model_selection as sk
import numpy as np

def numberOfAuthors(df):
    for x in range(0, len(df)):
        if type(df['authors'][x]) == float:
            df['authors'][x] = 0
        else:
            result = re.findall(r",", df['authors'][x])
            df['authors'][x] = len(result) + 1
    # print(df)
    return df

def predictByAuthors(df):
    #prepare data
    df = numberOfAuthors(df)

    x = df['authors'].to_numpy()
    y = df['type'].to_numpy()

    x_train, x_val, y_train, y_val = sk.train_test_split(x, y, test_size=0.2, random_state=0)

    authorMod = LogisticRegression()

    #reshaping x_data
    x_train = np.reshape(x_train, (-1, 1))
    x_val = np.reshape(x_val, (-1, 1))
    print("shape of x_val: ", x_val.shape)
    print(x_val)

    #ensuring correct type within y_data
    y_train = y_train.astype('int')
    y_val = y_val.astype('int')
    print("shape of y_val: ", y_val.shape)
    print(y_val)


    authorMod = authorMod.fit(x_train, y_train)

    y_pred = authorMod.predict(x_val)
    print(y_pred)
    print(y_val)
    y_pred = y_val.astype('int')

    print("Accuracy: ", metrics.accuracy_score(y_val, y_pred))
    print("Recall: ", metrics.recall_score(y_val, y_pred))
    print("Precision: ", metrics.recall_score(y_val, y_pred))
    print("F1-score: ", metrics.f1_score(y_val, y_pred))

    return