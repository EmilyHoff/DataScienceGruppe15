import pandas as pd
import numpy as np
import sklearn
import re
import math
import wandb

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics as metr

from operator import itemgetter


def naive_bayes_authors(df):

    x = df['authors'].to_numpy()
    y = df['type'].to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

    x_train = np.reshape(x_train, (-1,1))
    x_val = np.reshape(x_val, (-1,1))
    print(x_train.shape)
    print(x_val.shape)

    y_train = y_train.astype('int')
    y_val = y_val.astype('int')
    print(y_train.shape)
    print(y_val.shape)

    multiNB = MultinomialNB()
    gaussNB = GaussianNB()
    bernNB = BernoulliNB()

    multiNB.fit(x_train, y_train)
    gaussNB.fit(x_train, y_train)
    bernNB.fit(x_train, y_train)

    y_pred_multi = multiNB.predict(x_val)
    y_pred_gauss = gaussNB.predict(x_val)
    y_pred_bern = bernNB.predict(x_val)


    multinb_accuracy = metr.accuracy_score(y_val, y_pred_multi)
    multinb_conf_matr = metr.confusion_matrix(y_val, y_pred_multi)
    multinb_f1_score = metr.f1_score(y_val, y_pred_multi)
    print('multi accuracy:', multinb_accuracy)
    print('multi confusion matrix:', multinb_conf_matr)
    print('multi f1 score:', multinb_f1_score)

    gaussnb_acc = metr.accuracy_score(y_val, y_pred_gauss)
    gaussnb_conf_matr = metr.confusion_matrix(y_val, y_pred_gauss)
    gaussnb_f1_score = metr.f1_score(y_val, y_pred_gauss)
    print('gauss accuracy:', gaussnb_acc)
    print('gauss confusion matrix:', gaussnb_conf_matr)
    print('gauss f1 score:', gaussnb_f1_score)

    bernnb_accuracy = metr.accuracy_score(y_val, y_pred_bern)
    bernnb_conf_matr = metr.confusion_matrix(y_val, y_pred_bern)
    bernnb_f1_score = metr.f1_score(y_val, y_pred_bern)
    print('bern accuracy:', bernnb_accuracy)
    print('bern confusion matrix:', bernnb_conf_matr)
    print('bern f1 score:', bernnb_f1_score)
    return

def naive_bayes_content(df, config):

    print('nb model')

    max_len = max([len(seq) for seq in df["Article encoded"]])
    max_len = 250
    padded_list = [seq + [0.0] * (max_len - len(seq)) for seq in df["Article encoded"]]
    padded_list = [x[:max_len] for x in padded_list]
    newX = []

    for li in padded_list:
        oneInput = []
        for x in li:
            if isinstance(x,list):
                oneInput.append(x[0])
            else:
                oneInput.append(x)
        newX.append(oneInput)

    print('reshaping')
    newX = np.array(newX)
    newX = newX.reshape((newX.shape[0], -1))

    # normalize the data, so as we don't have negative values
    for i in range(0, len(newX)):
        newX[i] = (newX[i]-min(newX[i]))/(max(newX[i])-min(newX[i]))
    print(newX.shape)
    label = df['type']

    # print('train test split')
    acc = []
    pres = []
    rec = []
    f1 = []
    # for i in range(0, 50):
    X_train, X_val, y_train, y_val = train_test_split(newX, label, test_size=0.2, random_state=0)

    y_train = y_train.astype('int')
    y_val = y_val.astype('int')

    print(y_train.shape)
    print(y_val.shape)

    nb = ComplementNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_val)

    nb_acc = metr.accuracy_score(y_val, y_pred)
    precision = metr.precision_score(y_val, y_pred, zero_division=1,average='weighted')
    recall = metr.recall_score(y_val, y_pred, zero_division=1,average='weighted')
    f1_score = metr.f1_score(y_val, y_pred, zero_division=1, average='weighted')
    # wandb.log({'accuracy': nb_acc, 'precision': precision, 'recall': recall, 'f1 score': f1_score})

    # acc.append((nb_acc, i))
    # pres.append((precision, i))
    # rec.append((recall, i))
    # f1.append((f1_score, i))

    conf_matr = metr.confusion_matrix(y_val, y_pred)
    class_report = metr.classification_report(y_val, y_pred, zero_division=1)

    # nb_acc = max(acc, key=itemgetter(1))[0]
    # precision = max(pres, key=itemgetter(1))[0]
    # recall = max(rec, key=itemgetter(1))[0]
    # f1_score = max(f1, key=itemgetter(1))[0]

    print('accuracy:{:0.5f}'.format(nb_acc)) # 0.65
    print('precision:{:0.5f}'.format(precision)) # 0.685
    print('recall:{:0.5f}'.format(recall)) # 0.65
    print('f1-score:{:0.5f}'.format(f1_score)) # 0.664

    print('confusion matrix:\n', conf_matr,'\n')
    print('report:\n', class_report)
    return nb_acc


sweep_configuration = {
    'method': 'bayes',
    'metric': {'name': 'accuracy', 'goal': 'maximize'},
    'parameters': {
        'max_iter': {'values': [1, 5, 10, 25, 50, 100, 250]},
        'random_state': {'min': 0, 'max': 50}
    }
}

def nb_wandbHandler (df):
    wandb.init(project='bayes run')
    accuracy = naive_bayes_content(df)
    wandb.log({'accuracy': accuracy})


# wandb.init()

# sweep_id = wandb.sweep(sweep=sweep_configuration, project='bayes run')
# wandb.agent(sweep_id=sweep_id, function=naiveBayesClassifier.naive_bayes_content(dfEncoded))

# wandb.finish()



