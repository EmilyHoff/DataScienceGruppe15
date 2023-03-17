from sklearn.linear_model import LinearRegression
from sklearn import metrics


def linMod(x_train, y_train, x_val, y_val):
    linMod = LinearRegression()

    linMod = linMod.fit(x_train, y_train)

    y_pred = linMod.predict(x_val)

    print("Accuracy: ", metrics.accuracy_score(y_val, y_pred))

    return linMod
