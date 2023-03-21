import pandas as pd
import numpy as np
import sklearn

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import metrics as metr


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
    multinb_r2_score = metr.r2_score(y_val, y_pred_multi)
    print('multi accuracy:', multinb_accuracy)
    print('multi confusion matrix:', multinb_conf_matr)
    print('multi r2 score:', multinb_r2_score)

    gaussnb_acc = metr.accuracy_score(y_val, y_pred_gauss)
    gaussnb_conf_matr = metr.confusion_matrix(y_val, y_pred_gauss)
    gaussnb_r2_score = metr.r2_score(y_val, y_pred_gauss)
    print('gauss accuracy:', gaussnb_acc)
    print('gauss confusion matrix:', gaussnb_conf_matr)
    print('gauss r2 score:', gaussnb_r2_score)

    bernnb_accuracy = metr.accuracy_score(y_val, y_pred_bern)
    bernnb_conf_matr = metr.confusion_matrix(y_val, y_pred_bern)
    bernnb_r2_score = metr.r2_score(y_val, y_pred_bern)
    print('bern accuracy:', bernnb_accuracy)
    print('bern confusion matrix:', bernnb_conf_matr)
    print('bern r2 score:', bernnb_r2_score)
    return

# def naive_bayes_content(df)




