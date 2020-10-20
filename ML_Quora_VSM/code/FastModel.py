import time
import pickle

import pandas as pd
from numpy import int8, linspace
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score, plot_confusion_matrix

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def TrainNB(dataset):
    X = dataset.preprocessed.values.ravel()
    y = dataset.target.values.ravel()
    start = time.time()
    sampler = None
    undersampler = None

    sampler = RandomOverSampler(sampling_strategy=0.15, random_state=6)
    #  undersampler = RandomUnderSampler(sampling_strategy=0.4, random_state=6)
    vectorizer = TfidfVectorizer(max_features=300000, ngram_range=(1, 4))
    model = BernoulliNB()

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

    print("Training model now")
    model.fit(X_tfidf, y)
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
    output.to_csv('FastSubmissions.csv', index=False)

    print("Local testing")
    ideal_test = pd.read_csv('../dataset/ideal_40.csv').target_y.astype(int8)
    print(f"Local F1 score: {f1_score(ideal_test, predictions)}")
    plot_confusion_matrix(model, X_tfidf, ideal_test, display_labels=['sincere', 'insincere'])
    plt.ticklabel_format = 'plain'
    plt.savefig('Fastmodel_CM.png')
    plt.show()

if __name__ == "__main__":
    start = time.time()
    train_file = "preprocessed_train.csv"
    test_file = "preprocessed_test.csv"

    traind = pd.read_csv(train_file)
    traind.fillna('', inplace=True)
    traind.target = traind.target.astype(int8)
    print("Datasets loaded")

    vectorizer, model = TrainNB(traind)
    now = time.time()
    print(f"Overall time: {now-start:.2f}s")
    start=now

    pickle.dump(model, open('FastModel.sav', 'wb'))
    pickle.dump(vectorizer, open('Vectorizer.sav', 'wb'))

    testd = pd.read_csv(test_file)
    testd.fillna('', inplace=True)
    Predict(vectorizer, model, testd)
    print("Predictions successfully written back to file\n")

    with open("TFIDF_words.txt", "w") as outputFile:
        for word in vectorizer.get_feature_names():
            outputFile.write("%s\n" % word)

    print("TFIDF features written to file")
