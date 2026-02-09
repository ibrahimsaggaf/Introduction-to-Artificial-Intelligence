import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def split(X, y):
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.7, shuffle=True, random_state=1111
    )

    return train_X, test_X, train_y, test_y


def scale(X):
    scaler = StandardScaler()

    return scaler.fit_transform(X)


def feature_selection(features, target, top):
    model = RandomForestRegressor()
    model.fit(features, target)
    importance = model.feature_importances_
    idxs = np.argsort(importance)[::-1]

    return features[:, idxs[:top]]


def performance(target, prediction):

    return mean_squared_error(target, prediction)