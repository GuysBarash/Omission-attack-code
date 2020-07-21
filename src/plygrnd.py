import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn.svm import SVC
import sklearn
import tqdm
import arff
import re

import os
from multiprocessing.pool import Pool
import shutil

from tools.Logger import Logger
from tools.paths import itemized as paths
from tools.params import itemized as params
from sklearn import svm

import nltk
from nltk.corpus import stopwords  # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    # nltk.download('stopwords')
    pass


def reviewWords(review):
    data_train_Exclude_tags = re.sub(r'<[^<>]+>', " ", review)  # Excluding the html tags
    data_train_num = re.sub(r'[0-9]+', 'number', data_train_Exclude_tags)  # Converting numbers to "NUMBER"
    data_train_lower = data_train_num.lower()  # Converting to lower case.
    data_train_split = data_train_lower.split()  # Splitting into individual words.
    stopWords = set(stopwords.words("english"))

    meaningful_words = [w for w in data_train_split if not w in stopWords]  # Removing stop words.

    return (" ".join(meaningful_words))


if __name__ == '__main__':
    train_size = 10000
    test_size = 100
    input_path = r"C:\school\thesis\omission\imdb\IMDB Dataset.csv"
    df = pd.read_csv(input_path)
    df['sentiment'] = df['sentiment'] == 'positive'
    df = df.sample(n=(train_size + test_size))

    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=4000)

    forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
    data_train_features = vectorizer.fit_transform(df_train['review'])

    print("Training the classifier\n")
    forest = forest.fit(data_train_features, df_train["sentiment"])
    score = forest.score(data_train_features, df_train["sentiment"])
    print("Mean Accuracy of the Random forest is: %f" % (score))

    print("Training the classifier on test set\n")
    data_train_features = vectorizer.fit_transform(df_test['review'])
    score = forest.score(data_train_features, df_test["sentiment"])
    print("Mean Accuracy of the Random forest is: %f" % (score))
