from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import extract_data
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


def vectorize(data : pd.DataFrame, ngram_range : tuple = (1, 3)):
    vec = TfidfVectorizer(ngram_range=ngram_range,
                          min_df=3, max_df=0.9, strip_accents='unicode', use_idf=True,
                          analyzer='word',
                          stop_words='english',
                          smooth_idf=True, sublinear_tf=True,
                          max_features=2000)

    return vec.fit_transform(data['comment_text'])

def logistic_regression(data : pd.DataFrame):
    class_names = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
    X = vectorize(data)
    X_train, X_test, y_train, y_test = train_test_split(X, np.array([row.tolist() for row in data[class_names].to_numpy()]))

    log_reg = LogisticRegression(C=0.1, solver='sag', max_iter=1000)
    classifier = MultiOutputRegressor(log_reg)

    cv_score = np.mean(cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=KFold(n_splits=10), n_jobs=-1))
    print(f'Cross accuracy score: {cv_score}')

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_train)
    ac_score = accuracy_score(y_train, y_pred)
    print(f'Training accuracy score: {ac_score}')

    y_pred = classifier.predict(X_test)
    ac_score = accuracy_score(y_test, y_pred)
    print(f'Testing accuracy score: {ac_score}')

    print('------------------------------')


if __name__ == "__main__":
    train = extract_data.extract("train.csv")
    logistic_regression(train)
