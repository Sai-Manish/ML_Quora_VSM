#!/home/shubhayu/anaconda3/bin/python
# coding: utf-8

import time
import pickle
from pprint import pprint

import pandas as pd
from numpy import int8, logspace
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, plot_confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def Train(dataset, params={}):
    X = dataset.preprocessed.values.ravel()
    y = dataset.target.values.ravel()
    start = time.time()

    # Relevant line
    sampler = None
    undersampler = None

    sampler = SMOTE(sampling_strategy=0.15, k_neighbors=500, random_state=6)
    undersampler = RandomUnderSampler(sampling_strategy=0.2, random_state=6)

    vectorizer = TfidfVectorizer(max_features=250000, ngram_range=(1, 3))

    model = SGDClassifier(
                        alpha=1e-6,
                        loss='log',
                        penalty='elasticnet',
                        validation_fraction=0.1,
                        l1_ratio=0.5,
                        n_iter_no_change=100,
                        n_jobs=-1,
                        random_state=42
    )

    X_tfidf = vectorizer.fit_transform(X)
    print("TFIDF shape: ", X_tfidf.shape)
    print(f"Vectorizing took: {time.time() - start:.4f}s")
    start = time.time()

    if sampler:
        X_tfidf, y = sampler.fit_resample(X_tfidf, y)
        print(f"Up sampler done: {X_tfidf.shape}")
        print(f"Upsampler took: {time.time() - start:.4f}s.")
        start = time.time()

    if undersampler:
        X_tfidf, y = undersampler.fit_resample(X_tfidf, y)
        print(f"Down sampler done: {X_tfidf.shape}")
        print(f"Downsampler took: {time.time() - start:.4f}s.")
        start = time.time()

    print("Training SGD regression model now")
    model.fit(X_tfidf, y)
    print(model)

    pickle.dump(model, open('models/SGDModel.sav', 'wb'))
    print(f"Model training took: {time.time() - start:.4f}s")

    return vectorizer, model

def Predict(vectorizer, model, dataset):
    ids = dataset.qid.values.ravel()
    X = dataset.preprocessed.values.ravel()

    X_tfidf = vectorizer.transform(X)

    print("Running predictions now")
    predictions = model.predict(X_tfidf)

    output = pd.DataFrame([ids, predictions], ['qid', 'target']).transpose()
    assert output.shape == (522449, 2)

    print("Predictions written back to submission file")
    output.to_csv('submission/SGD_Submissions.csv', index=False)

    print("Local testing")
    ideal_test = pd.read_csv('../dataset/ideal_40.csv').target.astype(int8)
    print(f"Local F1 score: {f1_score(ideal_test, predictions):.5f}")
    plot_confusion_matrix(model, X_tfidf, ideal_test, display_labels=['sincere', 'insincere'])
    plt.ticklabel_format = 'plain'
    plt.show()
    plt.savefig('confusion_matrices/sgd_model_CM.png')

if __name__ == "__main__":
    start = time.time()
    train_file = "../processed/preprocessed_train.csv"
    test_file = "../processed/preprocessed_test.csv"

    traind = pd.read_csv(train_file)
    traind.fillna('', inplace=True)
    traind.target = traind.target.astype(int8)
    print("Datasets loaded")

    vectorizer, model = Train(traind)
    now = time.time()
    print(f"Overall time: {now-start:.2f}s")
    start=now

    testd = pd.read_csv(test_file)
    testd.fillna('', inplace=True)
    Predict(vectorizer, model, testd)
    print("Predictions successfully written back to file\n")
