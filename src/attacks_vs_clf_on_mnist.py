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

import sklearn.metrics as metrics
import tqdm

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

        callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, monitor='loss')]
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


def attack_KNN(S, sa, budget, src_label, trgt_label, clf_tag, datacols, labelcol, infosr):
    X = S[datacols]
    y = S[labelcol]
    S['Distance'] = get_distance(X, sa)
    S = S.sort_values(by=['Distance'])
    g = S.groupby(labelcol)

    Sk = g.get_group(src_label).iloc[:budget]
    Sk_indexes = Sk.index
    Stag = S.drop(index=Sk_indexes)
    return Stag


def attack_greedy(S, sa, budget, src_label, trgt_label, clf_tag, datacols, labelcol, infosr):
    Stag = S.copy()
    Stag['score'] = 0.0
    X = S[datacols]
    y = S[labelcol]
    Sk_indexes = list()
    r = pd.DataFrame(columns=datacols, index=[0])
    r.iloc[0] = sa
    sa = r

    current_prediction_of_target = infosr['prob_of_adv_for_TRGT_before_attack']
    sub_S = Stag[Stag[labelcol] == src_label]
    for k in tqdm.tqdm(range(budget), desc="Greedy attack selection", position=0, leave=True):
        scores = pd.Series(index=sub_S.index)
        for sidx, s in tqdm.tqdm(sub_S.iterrows(), total=sub_S.shape[0],
                                 desc=f"Round {k +1}/{budget}\t Prob of target: {current_prediction_of_target:>.4f}",
                                 position=0,
                                 leave=True):
            clf = get_clf(clf_tag)
            St = Stag.drop(sidx)
            clf = clf_fit(clf, clf_tag, St[datacols], St[labelcol])
            yhat = clf_predict_proba(clf, clf_tag, sa)
            # res = clf.predict_proba(sa)
            yhat = yhat[0][trgt_label]
            scores[sidx] = yhat
        current_prediction_of_target = scores.max()
        idx_to_remove = scores.idxmax()
        # print(f"Dropping {k+1}/{budget}: Score: {scores[idx_to_remove]}\tShape: {sub_S.shape}")
        Sk_indexes.append(idx_to_remove)
        sub_S = sub_S.drop(idx_to_remove)
    Stag = S.drop(index=Sk_indexes)
    return Stag


def attack_genetic(S, sa, budget, src_label, trgt_label, clf_tag, datacols, labelcol, infosr):
    ## parameters of genetic attack
    set_size = budget
    max_value_in_set = S.shape[0]
    legal_options = np.array(list(S.index))
    random.seed(64)
    number_of_generations = 500
    population_size = 200
    chance_to_mutate_idx = 3.0 / set_size
    number_of_offsprings = int(population_size / 2)
    cxpb = 0.3
    mutpb = 0.7
    verbose = True

    X = S[datacols]
    y = S[labelcol]
    S['Distance'] = get_distance(X, sa)

    r = pd.DataFrame(columns=datacols, index=[0])
    r.iloc[0] = sa
    sa = r

    def generate_random_create(S_distance, budget, src_label, datacols, labelcol, idx=0):
        St = S_distance.sort_values(by=['Distance'])
        St = St[St[labelcol] == src_label]
        St = St.iloc[idx * budget:(idx + 1) * budget]

        ind = set(St.index)
        ind = creator.Individual(ind)
        return ind

    # # To assure reproductibility, the RNG seed is set prior to the items
    # random.seed(64)

    creator.create("FitnessMax", base.Fitness, weights=(+1.0,))
    creator.create("Individual", set, n=10, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_item", random.randrange, max_value_in_set)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_item, set_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(p):
        # score = np.abs(np.array(list(individual)) - 100).mean()
        individual, datacols, labelcol = p
        idx_to_remove = list(individual)
        clf = get_clf(clf_tag)
        St = S.drop(idx_to_remove)

        clf = clf_fit(clf, clf_tag, St[datacols], St[labelcol])
        # _ = clf.fit(St[['X', 'Y']], St['C'])
        # res = clf.predict_proba(sa)
        res = clf_predict_proba(clf, clf_tag, sa)
        score = res[0][trgt_label]

        return score,

    def cxSet(ind1, ind2):
        pool = set.union(ind1, ind2)
        ind1 = creator.Individual(set(random.sample(pool, set_size)))
        ind2 = creator.Individual(set(random.sample(pool, set_size)))
        if len(ind1) != set_size or len(ind2) != set_size:
            j = 3
        return ind1, ind2

    def mutSet(individual):
        origin_creature = np.array(list(individual))
        random_creature = np.random.choice(np.setdiff1d(legal_options, origin_creature), size=set_size, replace=False)
        random_map = np.random.choice([0, 1], p=[1 - chance_to_mutate_idx, chance_to_mutate_idx], size=set_size)
        inv_random_map = 1 - random_map

        new_creature = (inv_random_map * origin_creature) + (random_map * random_creature)
        ind = creator.Individual(set(new_creature))
        return ind,

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", cxSet)
    toolbox.register("mutate", mutSet)
    toolbox.register("select", tools.selNSGA2)

    population = [generate_random_create(S, budget, src_label, datacols, labelcol, idx)
                  for idx in range(int(population_size / budget))]  # toolbox.population(n=population_size)
    halloffame = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("max", np.max, axis=0)

    # algorithms.eaMuPlusLambda(pop, toolbox, population_size, LAMBDA, CXPB, MUTPB, number_of_generations, stats,
    #                           halloffame=hof)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [(ind, datacols, labelcol) for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    progess = zip(invalid_ind, fitnesses)
    progess = tqdm.tqdm(progess, desc='Gen 0', total=len(invalid_ind))
    for ind, fit in progess:
        ind[0].fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, number_of_generations + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, number_of_offsprings, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [(ind, datacols, labelcol) for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        progess = zip(invalid_ind, fitnesses)
        progess = tqdm.tqdm(progess, desc=f'Gen {gen}/{number_of_generations}', total=len(invalid_ind))
        for ind, fit in progess:
            ind[0].fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, population_size)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if logbook[-1]['max'] > 0.55:
            break

    winner_creature = halloffame.items[0]
    Sk_indexes = list(winner_creature)
    Stag = S.drop(index=Sk_indexes)
    return Stag


def func(info):
    config_stuff = True
    if config_stuff:
        clear_folder(info['Outpath'])
        colors = {info['TRGT idx']: 'blue', info['SRC idx']: 'red', 2: 'yellow'}
        markers = {info['TRGT idx']: 'o', info['SRC idx']: 'o', 2: 'X'}
        labels = {info['TRGT idx']: 'TRGT', info['SRC idx']: 'SRC', 2: 'ADV'}
        trgt, src = np.random.choice(range(10), size=2, replace=False)
        labels_real_number = {'TRGT': trgt, 'SRC': src, 'adv': 999}
        infosr = pd.Series(name=info['Sig'])
        add_plot = info['PLOT']
        infosr['SRC number'] = src
        infosr['TRGT number'] = trgt
        infosr['Start time'] = datetime.datetime.now()

        msg = ''
        msg += '#######################' + "\n"
        msg += '#######################' + "\n"
        msg += f'## ATTACK : {info["Attack"]:>8} ##' + "\n"
        msg += f'## CLF    : {info["clf"]:>8} ##' + "\n"
        msg += f'## UID    : {info["uid"]:>8} ##' + "\n"
        msg += f'## PLOT   : {str(info["PLOT"]):>8} ##' + "\n"
        msg += f'## {src:^6} --> {trgt:^6} ##' + "\n"
        msg += '#######################' + "\n"
        msg += '#######################'
        print(msg)

    generate_data = True
    if generate_data:
        print("Generating data")

        # Label 0 is TRGT \ BLUE
        # Label 1 is SRC \ RED

        csv_path = r"C:\school\thesis\omission\mnist\mnist_train.csv"
        print("loading data.")
        rawdf = pd.read_csv(csv_path)
        print("Data loaded.")
        clf_tag = info['clf']
        clf = get_clf(clf_tag, save_weights=True)

        # Run until a suitable adv is found
        for attempt_idx in range(9999):
            srcdf = rawdf[rawdf['label'] == src].sample(int(info['samples'] / 2.0))
            trgtdf = rawdf[rawdf['label'] == trgt].sample(int(info['samples'] / 2.0))
            srcdf['label'] = 1
            trgtdf['label'] = 0
            samples = pd.concat([srcdf, trgtdf])
            samples = samples.sample(frac=1)

            potential_adv_points = rawdf[rawdf['label'] == src].sample(250)
            potential_adv_points['label'] = 1
            datacols = samples.columns[1:]
            labelcol = samples.columns[0]

            print(f"Testing Adv point {attempt_idx + 1}")
            S = samples
            X, y = S[datacols], S[labelcol]
            clf = clf_fit(clf, clf_tag, X, y, verbose=False)

            # Choose adv point
            y_hat = clf_predict_proba(clf, clf_tag, potential_adv_points[datacols])
            y_hat = y_hat[y_hat[:, 1] > info['adv prob of src thresholds'][0]]
            y_hat = y_hat[y_hat[:, 1] < info['adv prob of src thresholds'][1]]
            if y_hat.shape[0] <= 0:
                print(f"BAD prediction, retry [phase 0][Attempt: {attempt_idx + 1:>3}]")
                continue

            adv_point = potential_adv_points.iloc[y_hat[:, 1].argmin()]
            adv_X = adv_point[datacols]
            y_hat = clf_predict_proba(clf, clf_tag, (adv_X,))
            before_attack_prediction = y_hat[0][info['TRGT idx']]
            before_attack_prediction_src = y_hat[0][info['SRC idx']]
            if (before_attack_prediction_src > info['adv prob of src thresholds'][0]) \
                    and (before_attack_prediction_src < info['adv prob of src thresholds'][1]) \
                    or (clf_tag == 'ANN'):
                print(f"GOOD prediction: {before_attack_prediction:>.5f}")
                infosr['prob_of_adv_for_TRGT_before_attack'] = before_attack_prediction
                break

            else:
                print(f"BAD prediction: {before_attack_prediction:>.5f} [phase 1]")

    run_clf_without_attack = True
    if run_clf_without_attack:

        adv_sample = adv_X.copy()
        adv_sample[labelcol] = 2
        samples = samples.append(adv_sample)
        samples = samples.reset_index(drop=True)

        plot_data = True
        add_plot = True
        if plot_data and add_plot:
            outpath = os.path.join(info['Outpath'], 'adv.png')
            plot_sample(adv_X, outpath, 'ADV sample')

    apply_attack = True
    if apply_attack:
        print("Attacking!")
        S = samples[samples[labelcol] != 2]
        sa = samples[samples[labelcol] == 2][datacols].iloc[0]
        if info['Attack'] == 'KNN':
            Shat = attack_KNN(S, sa, info['budget'], info['SRC idx'], info['TRGT idx'], info['clf'],
                              datacols, labelcol, infosr)
        elif info['Attack'] == 'Greedy':
            Shat = attack_greedy(S, sa, info['budget'], info['SRC idx'], info['TRGT idx'], info['clf'],
                                 datacols, labelcol, infosr)
        elif info['Attack'] == 'Genetic':
            Shat = attack_genetic(S, sa, info['budget'], info['SRC idx'], info['TRGT idx'], info['clf'],
                                  datacols, labelcol, infosr)
        else:
            raise Exception(f"BAD Attack method: {info['Attack']}")

    run_clf_after_attack = True
    if run_clf_after_attack:
        clf = get_clf(info['clf'])
        adv_X = samples[samples[labelcol] == 2][datacols]
        X, y = Shat[datacols], Shat[labelcol]
        clf = clf_fit(clf, clf_tag, X, y)
        # _ = clf.fit(X, y)
        # y_hat = clf.predict_proba(adv_X)
        y_hat = clf_predict_proba(clf, clf_tag, adv_X)
        # print("Prediction: {}".format(y_hat))
        infosr['prob_of_adv_for_TRGT_after_attack'] = y_hat[0][info['TRGT idx']]

    export_final_result = True
    if export_final_result:
        infosr['End time'] = datetime.datetime.now()
        infosr['Duration'] = infosr['End time'] - infosr['Start time']

        hours, remainder = divmod(infosr['Duration'].total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        msg = ''
        msg += '@' * 30 + '\n'
        msg += f'@@@   Before   : {infosr["prob_of_adv_for_TRGT_before_attack"]:>.4f}    @@@\n'
        msg += f'@@@   After    : {infosr["prob_of_adv_for_TRGT_after_attack"]:>.4f}    @@@\n'
        msg += f'@@@   Duration : {int(hours):>02}:{int(minutes):>02}:{int(seconds):>02}  @@@\n'
        msg += '@' * 30 + '\n'
        print(msg)

        outpath = os.path.join(info['Outpath'], 'results.csv')
        infodf = pd.DataFrame(columns=infosr.index)
        infodf = infodf.append(infosr)
        infodf.to_csv(outpath, header=True)


if __name__ == '__main__':
    root_path = r'C:\school\thesis\clf vs learner MNIST'

if __name__ == '__main__':
    # make inputs
    info = dict()
    info['Start time'] = datetime.datetime.now()
    info['uid'] = np.random.randint(2 ** 25)
    info['SRC idx'] = 1
    info['TRGT idx'] = 0
    info['clf'] = sys.argv[1] if len(sys.argv) > 1 else 'ANN'
    info['Attack'] = sys.argv[2] if len(sys.argv) > 1 else 'KNN'
    info['samples'] = 400
    info['budget'] = int(np.ceil(np.sqrt(info['samples'])))
    info['adv prob of src thresholds'] = (0.56, 0.95)
    info['PLOT'] = sys.argv[3] if len(sys.argv) > 3 else True

    if info['PLOT']:
        sig = f'{info["Attack"]}__{info["clf"]}__{info["Start time"].strftime("T%S%M%HT%d%b%yT")}__PLOT'
    else:
        sig = f'{info["Attack"]}__{info["clf"]}__{info["Start time"].strftime("T%S%M%HT%d%b%yT")}'
    info['Sig'] = sig
    info['Outpath'] = os.path.join(root_path, str(info['Sig']))

if __name__ == '__main__':
    func(info)
    print("CODE COMPLETED.")
