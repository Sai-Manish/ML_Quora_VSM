#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import re
import string
from collections import Counter

import pandas as pd
import swifter
import numpy as np

import nltk
# nltk.download("stopwords")
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, SnowballStemmer

def getRegexExpr():
    CHEMISTRY_KEYWORDS = ['CH3', 'CH2', 'CH', 'OH', 'COOH', 'OCH3', 'NH2', 'CHO', 'o3', 'h2', 'o2']
    
    chemistry_regex = r"\b\w*({})\w*\b".format("|".join(CHEMISTRY_KEYWORDS))
    math_regex = r"\w*(\[math\][\w\s(){}|+,?:;<>*&\\\-=^./%!∘$]*\[/*\\*math\])+\w*"
    website_regex = r"(?:(?:(?:https?|ftp)\s?):\/\/|\b(?:[a-z\d]+\.))(?:(?:[^\s()<>]+|\((?:[^\s()<>]+|(?:\([^\s()<>]+\)))?\))+(?:\((?:[^\s()<>]+|(?:\(?:[^\s()<>]+\)))?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))?"
    hex_regex = r"\b0x[\w*]+\b"
    num_regex = r"(\w{0,5}[0-9]{3,30}\w{0,5})"
    lnum_regex = r"(\w{0,5}[0-9]{30,1000}\w{0,5})"

    # TODO: To fix
    unit_regex = r"\w*\b[\d]{0,1000}[a-zA-Z]+\b\w*"

    return {
        "website": website_regex,
        "mathexpr": math_regex,
        "chemexpr": chemistry_regex,
        "hexexpr": hex_regex,
        "snumexpr": num_regex,
        "lnumexpr": lnum_regex,
    }

# TODO: to be implemented after finding a proper corpus
def SpellCorrection(data):
    dataset = data.copy(deep=True)

    return dataset

def preprocess(data):
    dataset = data.copy(deep=True)

    flags = re.IGNORECASE | re.DOTALL | re.UNICODE

    dataset['preprocessed'] = dataset.question_text
    dataset.drop(columns=['question_text'] , inplace=True)

    for replacement, regex in getRegexExpr().items():
        dataset.preprocessed = dataset.preprocessed.str.replace(regex, replacement, flags)

    dataset.preprocessed = dataset.preprocessed.str.replace(r'(\d+)', r' \1 ', flags)
    print("Regex filtering done")

    symbols = list(string.punctuation)
    symbols.remove("'")
    dataset.preprocessed = dataset.preprocessed.str.replace(r"({})+".format("|\\".join(symbols)), ' ', flags)

    dataset.preprocessed = dataset.preprocessed.swifter.allow_dask_on_strings(enable=True).apply(nltk.word_tokenize)

    dataset.preprocessed = dataset.preprocessed.swifter.allow_dask_on_strings(enable=True).apply(nltk.tag.pos_tag)

    def get_wordnet_pos(tag):
        if tag[0] == 'J':
            return wordnet.ADJ
        elif tag[0] == 'V':
            return wordnet.VERB
        elif tag[0] == 'R':
            return wordnet.ADV
        else:
            return wordnet.NOUN

    print("Allocation of tags")
    #  Bottleneck step. Do something about this
    dataset.preprocessed = dataset.preprocessed.apply(lambda x: [(word.lower(), get_wordnet_pos(pos_tag)) for (word, pos_tag) in
    x])

    return dataset

def Lemmatizer(data):
    dataset = data.copy(deep=True)

    print("Lemmatizing")
    lemmatizer = WordNetLemmatizer()
    dataset.preprocessed = dataset.preprocessed.apply(lambda x: [lemmatizer.lemmatize(word, tag) for word, tag in x])

    print("Snowball Stemming the words")
    stemmer =  SnowballStemmer("english")
    dataset.preprocessed = dataset.preprocessed.swifter.allow_dask_on_strings(enable=True).apply(lambda x:
    [stemmer.stem(word) for word in x])

    return dataset

if __name__ == "__main__":
    filenames = ["../dataset/train.csv", "../dataset/test.csv"]

    for filename in filenames:
        print(f"\nSaving to file: ./preprocessed_{filename.split('/')[-1]}")
        i = 0
        for chunk in pd.read_csv(filename, chunksize=2*10**5):
            start = time.time()
            i += 1
            print(f"\nChunk {i} dims: {chunk.shape}")
            traind = preprocess(chunk)
            traind = Lemmatizer(traind)
            if i == 1:
                traind.to_csv(f"./preprocessed_{filename.split('/')[-1]}", index=False, mode='a')
            else:
                traind.to_csv(f"./preprocessed_{filename.split('/')[-1]}", index=False, mode='a', header=False)

            end = time.time()
            print(f"Chunk {i} took {end-start:.2f}s")
