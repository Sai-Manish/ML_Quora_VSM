import os
import sys
import time
import re
import string

import pandas as pd
import swifter
import numpy as np

import nltk
import contractions
# nltk.download("stopwords")
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, SnowballStemmer

# Function to get a dict of regex function for filtering
# FIXME: update the regex expressions for improved performance
def getRegexExpr():
    CHEMISTRY_KEYWORDS = ['CH3', 'CH2', 'CH', 'OH', 'COOH', 'OCH3', 'NH2', 'CHO', 'o3', 'h2', 'o2']
    
    chemistry_regex = r"\b\w*({})\w*\b".format("|".join(CHEMISTRY_KEYWORDS))
    math_regex = r"\w*(\[math\][\w\s(){}|+,?:;<>*&\\\-=^./%!∘$]*\[/*\\*math\])+\w*"
    website_regex = r"(?:(?:(?:https?|ftp)\s?):\/\/|\b(?:[a-z\d]+\.))(?:(?:[^\s()<>]+|\((?:[^\s()<>]+|(?:\([^\s()<>]+\)))?\))+(?:\((?:[^\s()<>]+|(?:\(?:[^\s()<>]+\)))?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))?"
    hex_regex = r"\b0x[\w*]+\b"
    num_regex = r"(\w{0,5}[0-9]{3,30}\w{0,5})"
    lnum_regex = r"(\w{0,5}[0-9]{30,1000}\w{0,5})"

    return {
        "website": website_regex,
        "mathexpr": math_regex,
        "chemexpr": chemistry_regex,
        "": hex_regex,
        "snumexpr": num_regex,
        "": lnum_regex,
    }

# Helper function to rename the tags
def get_wordnet_pos(tag):
    if tag[0] == 'J':
        return wordnet.ADJ
    elif tag[0] == 'V':
        return wordnet.VERB
    elif tag[0] == 'R':
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(data):

    # Make a copy of the dataframe
    dataset = data.copy(deep=True)

    # Declare some flags for regex
    flags = re.IGNORECASE | re.DOTALL | re.UNICODE

    # Rename the column
    dataset = dataset.rename(columns={"question_text": "preprocessed"})

    # Run all the regex expressions one by one, on the entire dataset
    for replacement, regex in getRegexExpr().items():
      dataset.preprocessed = dataset.preprocessed.str.replace(regex, replacement, flags)

    # Separate digits from characters
    dataset.preprocessed = dataset.preprocessed.str.replace(r'(\d+)', r' \1 ', flags)
    print("Regex filtering done")
    
    # Expanding contraction words
    print("Removing some grammatical contractions")
    dataset.preprocessed = dataset.preprocessed.swifter.allow_dask_on_strings(enable=True).apply(contractions.fix)

    # Remove all symbols from the strings
    symbols = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
     '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
     '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
     '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
     '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
    dataset.preprocessed = dataset.preprocessed.str.replace(r"({})+".format("|\\".join(symbols)), ' ', flags)

    # Split sentences into an array of word tokens
    print("Tokening words using nltk")
    dataset.preprocessed = dataset.preprocessed.swifter.allow_dask_on_strings(enable=True).apply(nltk.word_tokenize)

    # Assign a part of speech positional tags to each of the words
    #  dataset.preprocessed = dataset.preprocessed.swifter.allow_dask_on_strings(enable=True).apply(nltk.tag.pos_tag)

    # Rename all the tags into useful format
    #  print("Allocation of tags")
    #  Bottleneck step. Do something about this
    #  dataset.preprocessed = dataset.preprocessed.apply(lambda x: [(word.lower(), get_wordnet_pos(pos_tag)) for (word, pos_tag) in
    #  x])

    return dataset

def Lemmatizer(data):
    dataset = data.copy(deep=True)

    # Apply the WordNet lemmatizer
    #  print("Lemmatizing")
    #  lemmatizer = WordNetLemmatizer()
    #  dataset.preprocessed = dataset.preprocessed.apply(lambda x: [lemmatizer.lemmatize(word, tag) for word, tag in x])

    # Apply Snowball stemming on all the English words
    print("Snowball Stemming the words")
    stemmer = SnowballStemmer("english")
    dataset.preprocessed = dataset.preprocessed.swifter.allow_dask_on_strings(enable=True).apply(lambda x: [stemmer.stem(word) for word in x])
    dataset.preprocessed = dataset.preprocessed.apply(" ".join)

    return dataset

if __name__ == "__main__":
    filenames = ["../dataset/train.csv", "../dataset/test.csv"] 

    for filename in filenames:
        outfile = "../processed/preprocessed_" + filename.split('/')[-1]
        print(f"\nSaving to file: {outfile}")
        i = 0
        for chunk in pd.read_csv(filename, chunksize=2e5):
            start = time.time()
            i += 1
            print(f"\nChunk {i} dims: {chunk.shape}")
            traind = preprocess(chunk)
            traind = Lemmatizer(traind)
            if i == 1:
                traind.to_csv(f"{outfile}", index=False, mode='w')
            else:
                traind.to_csv(f"{outfile}", index=False, mode='a', header=False)

            end = time.time()
            print(f"Chunk {i} took {end-start:.2f}s")
