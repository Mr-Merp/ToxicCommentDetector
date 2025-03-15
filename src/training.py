from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import extract_data
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


def vectorize(data : pd.DataFrame, ngram_range : tuple = (1, 6)):
    vec = TfidfVectorizer(ngram_range=ngram_range,
                          min_df=3, max_df=0.9, strip_accents='unicode', use_idf=True,
                          analyzer='word',
                          stop_words='english',
                          smooth_idf=True, sublinear_tf=True,
                          max_features=2000)

    return vec.fit_transform(data['comment_text'])


def multioutput_classification(data: pd.DataFrame):
    class_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    X = vectorize(data)
    X_train, X_test, y_train, y_test = train_test_split(X, np.array([row.tolist() for row in data[class_names].to_numpy()]))

    log_reg = LogisticRegression(C=0.1, solver='sag', max_iter=1000)
    classifier = MultiOutputClassifier(log_reg)

    cv_score = np.mean(
        cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=KFold(n_splits=10), n_jobs=-1))
    print(f'Cross accuracy score: {cv_score}')

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_train)
    ac_score = accuracy_score(y_train, y_pred)
    print(f'Training accuracy score: {ac_score}')

    y_pred = classifier.predict(X_test)
    ac_score = accuracy_score(y_test, y_pred)
    print(f'Testing accuracy score: {ac_score}')

    print('------------------------------')

def logistic_regression(data : pd.DataFrame) -> list[LogisticRegression]:
    X = vectorize(data)
    scores = []
    classifiers = []

    for class_name in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
        X_train, X_test, y_train, y_test = train_test_split(X, data[class_name])

        classifier = LogisticRegression(C=0.1, solver='sag', max_iter=1000)
        X_train_resampled, y_train_resampled = X_train, y_train
        if class_name != 'toxic':
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        cv_score = np.mean(cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=KFold(n_splits=10), n_jobs=-1))
        scores.append(cv_score)
        print(f'Tests for {class_name}')
        print(f'Cross accuracy score: {cv_score}')

        classifier.fit(X_train_resampled, y_train_resampled)
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

    return classifiers


if __name__ == "__main__":
    train = extract_data.extract("train.csv")
    logistic_regression(train)
