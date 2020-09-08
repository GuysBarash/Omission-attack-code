import datetime
import os
import tools.misc as misc
import sys
import csv
import time
import tqdm
import shutil
from copy import deepcopy

import tools.paths as paths
import tools.config as config
import tools.params as params

import numpy as np
import pandas as pd
import matplotlib

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random

from tensorflow import keras as K
from tensorflow.keras import layers
import tensorflow as tf

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import warnings
import sklearn.metrics as metrics
import tqdm
import time
from sklearn.datasets.samples_generator import make_blobs

UID = np.random.randint(2 ** 20)


def plot_sample(img, path, title=None):
    mat = img.values.reshape((28, 28))
    if title is not None:
        plt.title(title)
    imgplot = plt.imshow(mat)
    plt.savefig(path)
    plt.close()


def clf_fit(clf, clf_tag, X, y, verbose=False):
    if clf_tag == 'ANN':
        y0 = 1 - y
        y1 = y
        y2d = np.array([y0, y1]).T

        class haltCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if (logs.get('acc') >= 0.95):
                    # print("\n\n\nReached 0.8 accuracy value so cancelling training!\n\n\n")
                    self.model.stop_training = True

        trainingStopCallback = haltCallback()

        callbacks = list()  # [tf.keras.callbacks.EarlyStopping(patience=3, monitor='loss')]
        clf.load_weights(f"model_{UID}.h5")
        history = clf.fit(X, y2d, epochs=500, batch_size=65536, verbose=verbose, callbacks=callbacks)
    else:
        _ = clf.fit(X, y)
    return clf


def clf_predict_proba(clf, clf_tag, X):
    if clf_tag == 'ANN':
        if type(X) is pd.DataFrame:
            yhat = clf.predict(X.to_numpy(dtype=np.float64))

        elif type(X) is tuple:
            Xt = X[0].to_frame().T
            yhat = clf.predict(Xt)

        else:
            yhat = clf.predict(X)
    else:
        yhat = clf.predict_proba(X)
    return yhat


def clf_predict(clf, clf_tag, X):
    if clf_tag == 'ANN':
        if type(X) is pd.DataFrame:
            yhat = clf.predict(X.to_numpy(dtype=np.float64))
        else:
            yhat = clf.predict(X)
        yhat = yhat.argmax(axis=1)
    else:
        yhat = clf.predict(X)
    return yhat


def get_clf(clf_name, save_weights=False):
    if clf_name == 'SVM':
        clf = SVC(kernel="linear", probability=True)
    elif clf_name == 'DTree':
        clf = DecisionTreeClassifier()
    elif clf_name == 'KNN5':
        clf = KNeighborsClassifier(n_neighbors=5)
    elif clf_name == 'Gaussian_NB':
        clf = GaussianNB()
    elif clf_name == 'ANN':

        tf.keras.backend.clear_session()
        clf = K.Sequential(
            [
                layers.InputLayer(input_shape=(784,)),
                layers.Dense(16, activation="relu", name="Hidden1"),
                layers.Dense(2, activation='softmax', name="output"),
            ]
        )
        clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        if save_weights:
            clf.save_weights(f"model_{UID}.h5")
        # clf = MLPClassifier(hidden_layer_sizes=(4,))

    else:
        raise Exception(f"BAD CLF encountered: {clf_name}")
    return clf


def clear_folder(path):
    if os.path.exists(path):
        all_items_to_remove = [os.path.join(path, f) for f in os.listdir(path)]
        for item_to_remove in all_items_to_remove:
            if os.path.exists(item_to_remove) and not os.path.isdir(item_to_remove):
                os.remove(item_to_remove)
            else:
                shutil.rmtree(item_to_remove)

    if not os.path.exists(path):
        os.makedirs(path)


def get_distance(X, sa):
    r = X - sa
    r = np.square(r)
    r = r.sum(axis=1)
    r = np.sqrt(r)
    return r


def func(info, rawdf):
    config_stuff = True
    if config_stuff:
        trgt, src = np.random.choice(range(10), size=2, replace=False)
        info['Sig'] = '{}_{}_{}'.format(src, trgt, info['uid'])
        info['Outpath'] = os.path.join(root_path, info['Sig'])
        clear_folder(info['Outpath'])

        infosr = pd.Series(name=info['Sig'])
        infosr['SRC number'] = src
        infosr['TRGT number'] = trgt
        infosr['Start time'] = datetime.datetime.now()

        msg = ''
        msg += '#######################' + "\n"
        msg += '#######################' + "\n"
        msg += f'## CLF    : {info["clf"]:>8} ##' + "\n"
        msg += f'## UID    : {info["uid"]:>8} ##' + "\n"
        msg += f'## PLOT   : {str(info["PLOT"]):>8} ##' + "\n"
        msg += f'## {src:^6} --> {trgt:^6} ##' + "\n"
        msg += '#######################' + "\n"
        msg += '#######################'
        # print(msg)

    generate_data = True
    if generate_data:
        # print("Generating data")

        # Label 0 is TRGT \ BLUE
        # Label 1 is SRC \ RED

        # print("Data loaded.")
        clf_tag = info['clf']
        clf = get_clf(clf_tag, save_weights=True)

        traindf = rawdf.sample(600)
        testdf = rawdf.sample(500)

        datacols = traindf.columns[1:]
        labelcol = traindf.columns[0]

        S_train = traindf
        S_test = testdf
        X_train, y_train = S_train[datacols], S_train[labelcol]
        X_test, y_test = S_test[datacols], S_test[labelcol]
        clf = clf_fit(clf, clf_tag, X_train, y_train, verbose=False)
        y_hat_train = clf_predict(clf, clf_tag, X_train)
        y_hat_test = clf_predict(clf, clf_tag, X_test)

        train_accuracy = metrics.accuracy_score(y_train, y_hat_train)
        test_accuracy = metrics.accuracy_score(y_test, y_hat_test)

        # print(f"Train acc: {train_accuracy:>.3f}\tTest acc {test_accuracy:>.3f}")
        return train_accuracy, test_accuracy


if __name__ == '__main__':
    root_path = r'C:\school\thesis\clf compare MNIST'

if __name__ == '__main__':
    iterations = 50
    infosr = pd.Series(index=range(iterations))
    csv_path = r"C:\school\thesis\omission\mnist\mnist_train.csv"
    # print("loading data.")
    rawdf = pd.read_csv(csv_path)
    for clf_tag in ['ANN', 'DTree', 'KNN5', 'Gaussian_NB', 'SVM']:
        for idx in tqdm.tqdm(range(iterations), desc=f'Clf: {clf_tag}'):
            # make inputs
            info = dict()
            info['Start time'] = datetime.datetime.now()
            info['uid'] = np.random.randint(2 ** 25)
            info['clf'] = clf_tag
            info['samples'] = 400
            info['budget'] = int(np.ceil(np.sqrt(info['samples'])))
            info['PLOT'] = sys.argv[3] if len(sys.argv) > 3 else True

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_accuracy, test_accuracy = func(info, rawdf)
            infosr[idx] = test_accuracy

        time.sleep(0.2)
        print(f"CLF: {info['clf']}\t ACC {infosr.mean():>.3f}+{infosr.std():>.3f}")
        print("")
        time.sleep(0.2)
