from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import extract_data
import matplotlib.pyplot as plt
import seaborn as sns

def vectorize(data : pd.DataFrame, ngram_range : tuple = (1, 4)):
    vec = TfidfVectorizer(ngram_range=ngram_range,
                          min_df=3, max_df=0.9, strip_accents='unicode', use_idf=True,
                          analyzer='word',
                          stop_words='english',
                          smooth_idf=True, sublinear_tf=True,
                          max_features=2000)

    return vec.fit_transform(data['comment_text'])

def logistic_regression(data : pd.DataFrame) -> LogisticRegression:
    X = vectorize(data)
    X_train, X_test = X[0:100000], X[100000:153165]
    scores = []

    for class_name in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
        y_train, y_test = data[class_name].iloc[0:100000], data[class_name].iloc[100000:153165]
        classifiers = []
        classifier = LogisticRegression(C=0.1, solver='sag', max_iter=1000)

        cv_score = np.mean(cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=KFold(n_splits=10), n_jobs=-1))
        scores.append(cv_score)
        print(f'Tests for {class_name}')
        print(f'Cross accuracy score: {cv_score}')

        classifier.fit(X_train, y_train)
        classifiers.append(classifier)

        y_pred = classifier.predict(X_train)
        ac_score = accuracy_score(y_train, y_pred)
        print(f'Training accuracy score: {ac_score}')

        y_pred = classifier.predict(X_test)
        ac_score = accuracy_score(y_test, y_pred)
        print(f'Testing accuracy score: {ac_score}')

        print('------------------------------')

        cm = confusion_matrix(y_test, y_pred)
        print(f'Confusion Matrix for {class_name}:')
        print(cm)

        # Plot confusion matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-" + class_name, class_name],
                    yticklabels=["Non-" + class_name, class_name])
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title(f'Confusion Matrix for {class_name}')
        plt.show()

    return classifier

if __name__ == "__main__":
    train = extract_data.extract("train.csv")
    logistic_regression(train)
