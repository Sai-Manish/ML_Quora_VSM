#!/home/shubhayu/anaconda3/bin/python
# coding: utf-8

import time
import pickle
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, plot_confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def Train(dataset):
    X = dataset.preprocessed.values.ravel()
    y = dataset.target.values.ravel()
    
    # Start local clock
    start = time.time()

    # In case sampling is not being used
    sampler = None
    undersampler = None
    outlier_detect = None

    # Make objects of all the classes under use
    vectorizer = TfidfVectorizer(max_features=250000, ngram_range=(1, 3))
    sampler = SMOTE(sampling_strategy=0.15, k_neighbors=500, random_state=6)
    undersampler = RandomUnderSampler(sampling_strategy=0.2, random_state=6)

    #  outlier_detect = LocalOutlierFactor(n_neighbors=30, leaf_size=20, n_jobs=-1)
    model = LinearSVC(C=0.3, random_state=42, penalty='l1', dual=False, verbose=2)

    X_tfidf = vectorizer.fit_transform(X)
    print("TFIDF shape: ", X_tfidf.shape)
    print(f"Vectorizing took: {time.time() - start:.4f}s")
    start = time.time()

    del X
    gc.collect()

    if outlier_detect is not None:
        print("Starting outlier detection")
        outliers = outlier_detect.fit_predict(X_tfidf)
        indices = np.where(outliers==1)
        X_tfidf = X_tfidf[indices]
        y = y[indices]
        print(f"Number of outliers detected: {len(np.where(outliers==-1))}")
        print("TFIDF shape after outlier removal: ", X_tfidf.shape)
        print(f"Outlier detection and removal took: {time.time() - start:.4f}s")
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

    print("Training SVM model now")
    model.fit(X_tfidf, y)
    print(f"Model training took: {time.time() - start:.4f}s")

    return vectorizer, model 

def Predict(vectorizer, model, dataset):
    ids = dataset.qid.values.ravel()
    X = dataset.preprocessed.values.ravel()

    X_tfidf = vectorizer.transform(X)

    print("Running predictions now")
    #  scaled_probs = model.decision_function(X_tfidf)
    predictions = model.predict(X_tfidf)
    #  predictions = (scaled_probs - scaled_probs.min())/(scaled_probs.max() - scaled_probs.min())

    output = pd.DataFrame([ids, predictions], ['qid', 'target']).transpose()
    assert output.shape == (522449, 2)

    print("Predictions written back to submission file")
    output.to_csv('submission/LSVC_submissions.csv', index=False)

    print("Local testing")
    ideal_test = pd.read_csv('../dataset/ideal_40.csv').target.astype(np.int8)
    print(f"Local F1 score: {f1_score(ideal_test, predictions)}")
    plot_confusion_matrix(model, X_tfidf, ideal_test, display_labels=['sincere', 'insincere'])
    plt.ticklabel_format = 'plain'
    plt.savefig('confusion_matrices/linear_SVC_CM.png')
    plt.show()

if __name__ == "__main__":
    gc.enable()
    start = time.time()
    train_file = "../processed/preprocessed_train.csv"
    test_file = "../processed/preprocessed_test.csv"

    traind = pd.read_csv(train_file)
    traind.fillna('', inplace=True)
    traind.target = traind.target.astype(np.int8)
    print("Datasets loaded")

    vectorizer, model = Train(traind)
    now = time.time()
    print(f"Overall time: {now-start:.2f}s")
    start=now

    del traind
    gc.collect()

    pickle.dump(model, open('models/linear_SVC.sav', 'wb'))

    testd = pd.read_csv(test_file)
    testd.fillna('', inplace=True)
    Predict(vectorizer, model, testd)
    print("Predictions successfully written back to file\n")

    with open("TFIDF_words.txt", "w") as outputFile:
        for word in vectorizer.get_feature_names():
            outputFile.write("%s\n" % word)

    print("TFIDF features written to file")
