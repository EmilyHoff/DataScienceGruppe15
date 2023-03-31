import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn import metrics

def naive_bayes_content(encoded, labels, split):

    articles = np.array(encoded)
    labels = np.array(labels)

    X_train = articles[:split]
    y_train = labels[:split]
    X_val = articles[split:]
    y_val = labels[split:]

    y_train = y_train.astype('int')
    y_val = y_val.astype('int')

    nb = ComplementNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_val)

    accuracy = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred, zero_division=1,average='weighted')
    recall = metrics.recall_score(y_val, y_pred, zero_division=1,average='weighted')
    f1_score = metrics.f1_score(y_val, y_pred, zero_division=1, average='weighted')

    conf_matr = metrics.confusion_matrix(y_val, y_pred)
    class_report = metrics.classification_report(y_val, y_pred, zero_division=1)

    print('nb accuracy: {:0.5f}'.format(accuracy))
    print('nb precision: {:0.5f}'.format(precision))
    print('nb recall: {:0.5f}'.format(recall))
    print('nb f1-score: {:0.5f}'.format(f1_score))

    print('nb confusion matrix:\n', conf_matr,'\n')
    print('nb report:\n', class_report)

    return y_pred, nb