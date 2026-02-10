from sklearn.metrics import f1_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def split(X, y):
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.7, shuffle=True, stratify=y, random_state=1111
    )

    return train_X, test_X, train_y, test_y


def scale(X):
    scaler = StandardScaler()

    return scaler.fit_transform(X)


def performance(target, prediction, type):

    if type == 'supervised':
        return f1_score(target, prediction)
    
    elif type == 'unsupervised':
        return adjusted_rand_score(target, prediction)
    
    else:
        raise NotImplemented(
            'The selected learning type ({type}) is not implemented.'\
            ' Please choose one of the following types: supervised or unsupervised'
        )





