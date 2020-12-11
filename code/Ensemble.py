#!/home/shubhayu/anaconda3/bin/python
# coding: utf-8

import time
import pickle

import pandas as pd
from numpy import int8, linspace
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import f1_score, plot_confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import lightgbm as lgb

def Train(dataset):
    X = dataset.preprocessed.values.ravel()
    y = dataset.target.values.ravel()
    
    # Start local clock
    start = time.time()

    # In case sampling is not being used
    sampler = None
    undersampler = None

    # Make objects of all the classes under use
    vectorizer = TfidfVectorizer(max_features=250000, ngram_range=(1, 3), sublinear_tf=True)
    sampler = SMOTE(sampling_strategy=0.15, n_jobs=7, k_neighbors=500, random_state=6)
    undersampler = RandomUnderSampler(sampling_strategy=0.2, random_state=6)

    LGBMmodel = lgb.LGBMClassifier(n_estimators=600, subsample_for_bin=10000, reg_alpha=0.4,
    reg_lambda=0.1, subsample=0.9, learning_rate=0.05,
    silent=False, force_col_wise=True, colsample_bytree=0.8,
    random_state=6, objective='binary')

    LogModel = LogisticRegression(C=0.8, multi_class='ovr', random_state=42, penalty='l1', solver='liblinear',
    max_iter=1000, verbose=1)

    SVCmodel = LinearSVC(C=0.3, random_state=42, penalty='l1', max_iter=1000, dual=False, verbose=2)

    NBmodel = BernoulliNB()

    model = StackingClassifier([('nb', NBmodel), ('lsvc', SVCmodel), ('lgbm', LGBMmodel)], final_estimator=LogModel, cv=3, passthrough=True,
    n_jobs=3, verbose=2)

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
    
    print("Training an ensemble of models now")
    model.fit(X_tfidf, y)
    print(f"Model trainings took: {time.time() - start:.4f}s")

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
    output.to_csv('EnsembleSubmissions.csv', index=False)

    print("Local testing")
    ideal_test = pd.read_csv('../dataset/ideal_40.csv').target.astype(int8)
    print(f"Local F1 score: {f1_score(ideal_test, predictions):.6f}")
    plot_confusion_matrix(model, X_tfidf, ideal_test, display_labels=['sincere', 'insincere'])
    plt.ticklabel_format = 'plain'
    plt.savefig('ensemble_model_CM.png')
    plt.show()

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

    pickle.dump(model, open('Ensemble.sav', 'wb'))
    with open("TFIDF_words.txt", "w") as outputFile:
        for word in vectorizer.get_feature_names():
            outputFile.write("%s\n" % word)

    print("TFIDF features written to file")