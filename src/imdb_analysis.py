import numpy as np

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# -*- coding: utf-8 -*-
"""imdb.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17iVOtggpoPFDSmbdyCqsTpno5tP3ZMFN
"""

import os
from operator import itemgetter
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import multiprocessing as mp

warnings.filterwarnings('ignore')
# get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

import tensorflow as tf

from IPython.display import display, HTML

from keras.datasets import imdb
from keras.utils import np_utils, to_categorical
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

top_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=top_words)
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

X_train = pd.DataFrame(X_train)
data_cols = X_train.columns
label_cols = 'Label'

X_train[label_cols] = y_train
X_test = pd.DataFrame(X_test)
X_test[label_cols] = y_test

# # create the model [Emmbeding]
# model = Sequential()
# model.add(Embedding(top_words, 32, input_length=max_words))
# model.add(Flatten())
# model.add(Dense(250, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

# create the model [ Embedding + Conv ]
clf = Sequential()
clf.add(Embedding(top_words, 32, input_length=max_words))
clf.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
clf.add(MaxPooling1D(pool_size=2))
clf.add(Flatten())
clf.add(Dense(250, activation='relu'))
clf.add(Dense(1, activation='sigmoid'))
clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
clf.summary()

# Fit the model
start_time = datetime.now()
clf.fit(X_train[data_cols], X_train[label_cols], validation_data=(X_test[data_cols], X_test[label_cols]), epochs=2,
        batch_size=256, verbose=2)
# Final evaluation of the model
scores = clf.evaluate(X_test[data_cols], X_test[label_cols], verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
print(f"Duration: {datetime.now() - start_time}")

# Find an adverserial example to attack
attack_label = 1
res = clf.predict_proba(X_test[data_cols])
res = pd.DataFrame(res, index=X_test.index)
res['Prediction'] = clf.predict(X_test[data_cols])
res[label_cols] = X_test[label_cols]
res = res[res[label_cols] == attack_label]
res = res[(res['Prediction'] > 0.85) & (res['Prediction'] < 0.90)]
adv_idx = res.sample(1).index[0]
adv = pd.DataFrame(columns=X_test.columns)
adv.loc[0] = X_test.loc[adv_idx]
display(adv)

print(f"Original prediction: {adv.loc[0,label_cols]}")
prob = clf.predict(adv[data_cols])[0, 0]
print(f"Classification: {np.round(prob)} prob: {prob:>.3f}")

# Attack params
df = X_train
budget = int(np.ceil(np.sqrt(df.shape[0])))
mutation_rate = 2 * 1.0 / budget
offsprings = 300
parents = 3

per_round_winner = pd.DataFrame(
    columns=['Gen', 'creature_idx', 'prob_of_origin',
             'prob_of_target', 'change'] + [
                's_{}'.format(tk) for tk in range(budget)])
result_summary = pd.DataFrame(columns=['Gen', 'Best score'])
results_memory = pd.DataFrame(columns=['parent_idx', 'prob_of_origin', 'prob_of_target'])

# Choose initial parents
adv_point = adv.loc[0, data_cols]
k_dist = lambda p: ((p - adv_point) ** 2).sum()

creatures_dict = dict()

tparent = df.copy(deep=True)
tparent = tparent[tparent[label_cols] == 1.0]
tparent = tparent[data_cols]
tparent['k_dist'] = tparent.apply(k_dist, axis=1)
tparent = tparent.sort_values(by=['k_dist'], ascending=True)
tparent = tparent.iloc[0:parents * budget]
tparent = tparent.drop(columns=['k_dist'])
tparent = tparent.sort_index()

tparents_idxs = tparent.index
for cparent in range(parents):
    p = set(tparents_idxs[budget * cparent:budget * (cparent + 1)])
    hash_p = hash(str(p))
    creatures_dict[hash_p] = p

j = 3