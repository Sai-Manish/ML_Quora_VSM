import numpy as np
import scipy.sparse as sp
#  from sklearn.preprocessing import CountVectorizer, TfidfTransformer

from nltk.tag import pos_tag

class CustomTFIDF:
    def __init__(data, power, CountVector):
        self.CountVectorizer = CountVector
        self.words = None
        self.weights = {}
        self.power = power

    def _tagger(self):
        tagged = pos_tag(self.words)
        renamed_tags = map(lambda x: x[1][0], tagged)
        return renamed_tags

    # Function to compute the weights assigned to each word
    def _compute_weights(self):
        tagged_words = _tagger()


    # Function to scale the count vector and then multiply with POS weight
    def _scale(self, weights):
        pass

    # assign tags to the unique words and then convert to usable tags
    def _tag(self):
        pass

    # Actual exposed user interface
    def fit_transform(self, dataset):
        self.CountVectorizer.fit_transform(dataset)
        self.words = list(self.CountVectorizer.vocabulary_.keys())
        pass
