import time
import pickle
import gc

import pandas as pd
from numpy import int8, linspace
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
#  from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
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

    # Make objects of all the classes under use
    vectorizer = TfidfVectorizer(max_features=250000, ngram_range=(1, 3), sublinear_tf=True)
    sampler = SMOTE(sampling_strategy=0.15, n_jobs=7, k_neighbors=500, random_state=6)
    undersampler = RandomUnderSampler(sampling_strategy=0.2, random_state=6)

    SGD = SGDClassifier(
                        alpha=1e-6,
                        loss='log',
                        penalty='elasticnet',
                        validation_fraction=0.1,
                        l1_ratio=0.5,
                        n_iter_no_change=100,
                        n_jobs=-1,
                        random_state=42
    )

    SVCmodel = LinearSVC(C=0.3, random_state=42, penalty='l1', max_iter=1000, dual=False)

    BNBmodel = BernoulliNB(alpha=1e-7)

    models = [('lsvc', SVCmodel), ('sgd', SGD), ('bnb', BNBmodel)]
    model = VotingClassifier(models, voting='hard', n_jobs=3, verbose=True)

    X_tfidf = vectorizer.fit_transform(X)
    print("TFIDF shape: ", X_tfidf.shape)
    print(f"Vectorizing took: {time.time() - start:.4f}s")
    start = time.time()

    del X
    gc.collect()

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
    output.to_csv('submission/EnsembleSubmissions.csv', index=False)

    print("Local testing")
    ideal_test = pd.read_csv('../dataset/ideal_40.csv').target.astype(int8)
    print(f"Local F1 score: {f1_score(ideal_test, predictions):.6f}")
    plot_confusion_matrix(model, X_tfidf, ideal_test, display_labels=['sincere', 'insincere'])
    plt.ticklabel_format = 'plain'
    plt.show()
    plt.savefig('confusion_matrices/ensemble_model_CM.png')

if __name__ == "__main__":
    gc.enable()
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

    pickle.dump(model, open('models/Ensemble.sav', 'wb'))
