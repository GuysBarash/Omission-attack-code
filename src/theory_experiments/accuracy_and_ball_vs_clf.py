import datetime
import os
import sys
import csv
import time
import tqdm
import shutil
from copy import deepcopy
from tabulate import tabulate


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


def clf_fit(clf, clf_tag, X, y):
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
        clf.load_weights("model.h5")
        history = clf.fit(X, y2d, epochs=500, batch_size=65536, verbose=False, callbacks=callbacks)
    else:
        _ = clf.fit(X, y)
    return clf


def clf_predict_proba(clf, clf_tag, X):
    if clf_tag == 'ANN':
        if type(X) is pd.DataFrame:
            yhat = clf.predict(X.to_numpy(dtype=np.float64))
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

        clf = K.Sequential(
            [
                layers.InputLayer(input_shape=(2,)),
                layers.Dense(4, activation="relu", name="Hidden1"),
                layers.Dense(4, activation="relu", name="Hidden2"),
                layers.Dense(2, activation='softmax', name="output"),
            ]
        )
        clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        if save_weights:
            clf.save_weights('model.h5')
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


def attack_KNN(S, sa, budget, src_label, trgt_label, clf_tag):
    X = S[['X', 'Y']]
    y = S['C']
    S['Distance'] = get_distance(X, sa)
    S = S.sort_values(by=['Distance'])
    g = S.groupby('C')

    Sk = g.get_group(src_label).iloc[:budget]
    Sk_indexes = Sk.index
    Stag = S.drop(index=Sk_indexes)
    return Stag


def attack_greedy(S, sa, budget, src_label, trgt_label, clf_tag):
    Stag = S.copy()
    Stag['score'] = 0.0
    X = S[['X', 'Y']]
    y = S['C']
    Sk_indexes = list()
    r = pd.DataFrame(columns=['X', 'Y'], index=[0])
    r.iloc[0] = sa
    sa = r

    sub_S = Stag[Stag['C'] == src_label]
    for k in tqdm.tqdm(range(budget), desc="Greedy attack selection", position=0, leave=True):
        scores = pd.Series(index=sub_S.index)
        for sidx, s in tqdm.tqdm(sub_S.iterrows(), total=sub_S.shape[0], desc=f"Round {k +1}/{budget}", position=0,
                                 leave=True):
            clf = get_clf(clf_tag)
            St = Stag.drop(sidx)
            clf = clf_fit(clf, clf_tag, St[['X', 'Y']], St['C'])
            # _ = clf.fit(St[['X', 'Y']], St['C'])
            res = clf_predict_proba(clf, clf_tag, sa)
            # res = clf.predict_proba(sa)
            res = res[0][trgt_label]
            scores[sidx] = res
        idx_to_remove = scores.idxmax()
        # print(f"Dropping {k+1}/{budget}: Score: {scores[idx_to_remove]}\tShape: {sub_S.shape}")
        Sk_indexes.append(idx_to_remove)
        sub_S = sub_S.drop(idx_to_remove)
    Stag = S.drop(index=Sk_indexes)
    return Stag


def attack_genetic(S, sa, budget, src_label, trgt_label, clf_tag):
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

    X = S[['X', 'Y']]
    y = S['C']
    S['Distance'] = get_distance(X, sa)

    r = pd.DataFrame(columns=['X', 'Y'], index=[0])
    r.iloc[0] = sa
    sa = r

    def generate_random_create(S_distance, budget, src_label, idx=0):
        St = S_distance.sort_values(by=['Distance'])
        St = St[St['C'] == src_label]
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

    def eval_individual(individual):
        # score = np.abs(np.array(list(individual)) - 100).mean()

        idx_to_remove = list(individual)
        clf = get_clf(clf_tag)
        St = S.drop(idx_to_remove)

        clf = clf_fit(clf, clf_tag, St[['X', 'Y']], St['C'])
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

    population = [generate_random_create(S, budget, src_label, idx)
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
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

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
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

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
        infosr = pd.Series(name=info['Sig'])
        add_plot = info['PLOT']

        infosr['clf'] = info['clf']
        infosr['attack'] = info['Attack']

        msg = ''
        msg += '#######################' + "\n"
        msg += '#######################' + "\n"
        msg += f'## ATTACK : {info["Attack"]:>8} ##' + "\n"
        msg += f'## CLF    : {info["clf"]:>8} ##' + "\n"
        msg += f'## UID    : {info["uid"]:>8} ##' + "\n"
        msg += f'## PLOT   : {str(info["PLOT"]):>8} ##' + "\n"
        msg += '#######################' + "\n"
        msg += '#######################'
        print(msg)

    generate_data = True
    if generate_data:
        print("Generating data")

        # Label 0 is TRGT \ BLUE
        # Label 1 is SRC \ RED
        samples = pd.DataFrame(columns=['X', 'Y', 'C'], index=range(info['samples']))
        X, y = make_blobs(n_samples=info['samples'], n_features=2,
                          centers=info['samples center'], cluster_std=info['samples std'],
                          )
        samples['X'] = X[:, 0]
        samples['Y'] = X[:, 1]
        samples['C'] = y

        number_of_test_samples = int(0.2 * info['samples'])
        samples_test = pd.DataFrame(columns=['X', 'Y', 'C'], index=range(number_of_test_samples))
        X_test, y_test = make_blobs(n_samples=number_of_test_samples, n_features=2,
                                    centers=info['samples center'], cluster_std=info['samples std'],
                                    )
        samples_test['X'] = X_test[:, 0]
        samples_test['Y'] = X_test[:, 1]
        samples_test['C'] = y_test

    generate_adv_point = True
    if generate_adv_point:
        print("Generating Adv point")
        src_center = np.array(info['samples center'][info['SRC idx']])
        trgt_center = np.array(info['samples center'][info['TRGT idx']])

        src_weight = float(info['adv_dist'][info['SRC idx']])
        trgt_weight = float(info['adv_dist'][info['TRGT idx']])

        p_hardest = ((src_center * src_weight) + (trgt_center * trgt_weight)) / (src_weight + trgt_weight)
        mt = (src_center - trgt_center)
        mt = mt[1] / mt[0]
        line_slope = - 1 / mt
        line_offset = (p_hardest[1] - (line_slope * p_hardest[0]))
        difficulty = 1.0 - info['difficulty']
        max_x = +5.0
        p_easiest = np.array([max_x, line_slope * max_x + line_offset])
        adv_p = (p_easiest * difficulty) + (p_hardest * (1 - difficulty))
        adv_p_sr = pd.Series(index=['X', 'Y', 'C'], data=[adv_p[0], adv_p[1], 2], name=samples.iloc[-1].name + 1)
        samples = samples.append(adv_p_sr)

        plot_data = False
        if plot_data and add_plot:
            df_xport = samples
            fname = 'data_with_adv_raw'
            outpath = os.path.join(info['Outpath'], '{}.csv'.format(fname))
            df_xport.to_csv(outpath)

            outpath = os.path.join(info['Outpath'], '{}.png'.format(fname))
            grouped = df_xport.groupby('C')
            fig, ax = plt.subplots()
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            for key, group in grouped:
                group.plot(ax=ax, kind='scatter', x='X', y='Y', label=labels[key],
                           color=colors[key],
                           marker=markers[key])
            plt.savefig(outpath)
            plt.close()

    run_clf_without_attack = True
    if run_clf_without_attack:
        print("Testing Adv point")
        clf_tag = info['clf']
        clf = get_clf(clf_tag, save_weights=True)
        S = samples[samples['C'] != 2]
        adv_X = samples[samples['C'] == 2][['X', 'Y']]
        X, y = S[['X', 'Y']], S['C']
        clf = clf_fit(clf, clf_tag, X, y)
        # y_hat = clf.predict_proba(adv_X)
        y_hat = clf_predict_proba(clf, clf_tag, adv_X)
        # print("Prediction: {}".format(y_hat))
        infosr['prob_of_adv_for_TRGT_before_attack'] = y_hat[0][info['TRGT idx']]

        get_accuracy_on_test_before_attack = True
        if get_accuracy_on_test_before_attack:
            y_test_hat = clf_predict(clf, clf_tag, X_test)
            accuracy_score = metrics.accuracy_score(y_test, y_test_hat)
            infosr['accuracy_before_attack'] = accuracy_score
            infosr['1_minus_accuracy_before_attack'] = 1.0 - accuracy_score

        plot_data = False
        if plot_data and add_plot:
            outpath = os.path.join(info['Outpath'], 'clf_BASE.png')
            # score = clf.score(X, y)
            title = '{} - base'.format(info['clf'])
            ax = plt.subplot()
            fig, ax = plt.subplots()

            def make_meshgrid(x, y, h=.02):
                """Create a mesh of points to plot in

                Parameters
                ----------
                x: data to base x-axis meshgrid on
                y: data to base y-axis meshgrid on
                h: stepsize for meshgrid, optional

                Returns
                -------
                xx, yy : ndarray
                """
                x_min, x_max = -10, 10
                y_min, y_max = -10, 10
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))
                return xx, yy

            def plot_contours(ax, clf, xx, yy, **params):
                """Plot the decision boundaries for a classifier.

                Parameters
                ----------
                ax: matplotlib axes object
                clf: a classifier
                xx: meshgrid ndarray
                yy: meshgrid ndarray
                params: dictionary of params to pass to contourf, optional
                """
                # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = clf_predict(clf, clf_tag, np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                _ = params.pop('clf_tag')
                out = ax.contourf(xx, yy, Z, **params)
                return out

            X0, X1, y = X['X'], X['Y'], y
            xx, yy = make_meshgrid(X0, X1)
            plot_contours(ax, clf, xx, yy,
                          cmap=plt.cm.coolwarm, alpha=0.8, clf_tag=clf_tag)
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax.scatter(adv_X['X'], adv_X['Y'], marker=markers[2], color=colors[2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim((-10, +10))
            ax.set_ylim((-10, +10))
            ax.set_title(title)
            plt.savefig(outpath)
            plt.close('all')
            del outpath
            del ax, fig, xx, yy, title

    section_theory_calc_on_ball_near_adv_point = True
    if section_theory_calc_on_ball_near_adv_point:
        S = samples[samples['C'] != 2].copy()

        S_samples = S.shape[0]
        src_samples = S['C'].eq(info['SRC idx']).sum()
        trgt_samples = S['C'].eq(info['TRGT idx']).sum()
        other_samples = S_samples - src_samples - trgt_samples
        infosr['samples_in_dataset_total'] = S_samples
        infosr['samples_in_dataset_src'] = src_samples
        infosr['samples_in_dataset_trgt'] = trgt_samples
        infosr['samples_in_dataset_other'] = other_samples

        sa = samples[samples['C'] == 2][['X', 'Y']].iloc[0]
        S['Distance'] = get_distance(X, sa)
        S = S.sort_values(by=['Distance'], ascending=True)
        max_index_in_ball = S[S['C'].eq(info['SRC idx'])].iloc[info['budget']].name
        samples_in_ball = S.index.get_loc(max_index_in_ball)
        S = S.iloc[:samples_in_ball]

        S_samples = S.shape[0]
        src_samples = S['C'].eq(info['SRC idx']).sum()
        trgt_samples = S['C'].eq(info['TRGT idx']).sum()
        other_samples = S_samples - src_samples - trgt_samples
        infosr['samples_in_ball_total'] = S_samples
        infosr['samples_in_ball_src'] = src_samples
        infosr['samples_in_ball_trgt'] = trgt_samples
        infosr['samples_in_ball_other'] = other_samples

    apply_attack = False
    if apply_attack:
        print("Attacking!")
        S = samples[samples['C'] != 2]
        sa = samples[samples['C'] == 2][['X', 'Y']].iloc[0]
        if info['Attack'] == 'KNN':
            Shat = attack_KNN(S, sa, info['budget'], info['SRC idx'], info['TRGT idx'], info['clf'])
        elif info['Attack'] == 'Greedy':
            Shat = attack_greedy(S, sa, info['budget'], info['SRC idx'], info['TRGT idx'], info['clf'])
        elif info['Attack'] == 'Genetic':
            Shat = attack_genetic(S, sa, info['budget'], info['SRC idx'], info['TRGT idx'], info['clf'])
        else:
            raise Exception(f"BAD Attack method: {info['Attack']}")

    run_clf_after_attack = False
    if run_clf_after_attack:
        clf = get_clf(info['clf'])
        adv_X = samples[samples['C'] == 2][['X', 'Y']]
        X, y = Shat[['X', 'Y']], Shat['C']
        clf = clf_fit(clf, clf_tag, X, y)
        # _ = clf.fit(X, y)
        # y_hat = clf.predict_proba(adv_X)
        y_hat = clf_predict_proba(clf, clf_tag, adv_X)
        # print("Prediction: {}".format(y_hat))
        infosr['prob_of_adv_for_TRGT_after_attack'] = y_hat[0][info['TRGT idx']]

        get_accuracy_on_test_before_attack = True
        if get_accuracy_on_test_before_attack:
            y_test_hat = clf_predict(clf, clf_tag, X_test)
            accuracy_score = metrics.accuracy_score(y_test, y_test_hat)
            infosr['accuracy_after_attack'] = accuracy_score

        plot_data = True
        if plot_data and add_plot:
            outpath = os.path.join(info['Outpath'], 'clf_ATTACKED.png')
            # score = clf.score(X, y)
            title = '{} - ATTACKED'.format(info['clf'])
            ax = plt.subplot()
            fig, ax = plt.subplots()

            def make_meshgrid(x, y, h=.02):
                """Create a mesh of points to plot in

                Parameters
                ----------
                x: data to base x-axis meshgrid on
                y: data to base y-axis meshgrid on
                h: stepsize for meshgrid, optional

                Returns
                -------
                xx, yy : ndarray
                """
                x_min, x_max = -10, 10
                y_min, y_max = -10, 10
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))
                return xx, yy

            def plot_contours(ax, clf, xx, yy, **params):
                """Plot the decision boundaries for a classifier.

                Parameters
                ----------
                ax: matplotlib axes object
                clf: a classifier
                xx: meshgrid ndarray
                yy: meshgrid ndarray
                params: dictionary of params to pass to contourf, optional
                """
                # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = clf_predict(clf, clf_tag, np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                params.pop('clf_tag')
                out = ax.contourf(xx, yy, Z, **params)
                return out

            X0, X1, y = X['X'], X['Y'], y
            xx, yy = make_meshgrid(X0, X1)
            plot_contours(ax, clf, xx, yy,
                          cmap=plt.cm.coolwarm, alpha=0.8, clf_tag=clf_tag)
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax.scatter(adv_X['X'], adv_X['Y'], marker=markers[2], color=colors[2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim((-10, +10))
            ax.set_ylim((-10, +10))
            ax.set_title(title)
            plt.savefig(outpath)
            plt.close('all')
            del outpath
            del ax, fig, xx, yy, title

    export_final_result = True
    if export_final_result:
        outpath = os.path.join(info['Outpath'], 'results.csv')
        infodf = pd.DataFrame(columns=infosr.index)
        infodf = infodf.append(infosr)
        infodf = infodf.reindex(sorted(infodf.columns), axis=1)
        print(tabulate(infodf, headers='keys', tablefmt='psql'))
        infodf.to_csv(outpath, header=True)


def str2bool(s):
    return s.lower() in ['true', 't', 'y', 'yes', '1']


if __name__ == '__main__':
    root_path = r'C:\school\thesis\accuracy_and_ball_vs_clf'

if __name__ == '__main__':
    # make inputs
    # CLFs: SVM, DTree, KNN5, Gaussian_NB, ANN
    # Attacks: KNN, Greedy, Genetic

    info = dict()
    info['Start time'] = datetime.datetime.now()
    info['uid'] = np.random.randint(2 ** 25)
    info['SRC idx'] = 1
    info['TRGT idx'] = 0
    info['samples center'] = [(-3, -3), (+3, +3)]
    info['samples std'] = 3
    info['clf'] = sys.argv[1] if len(sys.argv) > 1 else 'ANN'
    info['Attack'] = sys.argv[2] if len(sys.argv) > 2 else 'KNN'
    info['adv_dist'] = (6, 10)
    info['samples'] = 400
    info['budget'] = int(np.ceil(np.sqrt(info['samples'])))
    info['difficulty'] = 0.3  # 0 is the easiest to attack
    info['PLOT'] = str2bool(sys.argv[3]) if len(sys.argv) > 3 else False
    info['run_id'] = sys.argv[4] if len(sys.argv) > 4 else 'X'

    if info['PLOT']:
        sig = f'{info["Attack"]}__{info["clf"]}__{info["Start time"].strftime("T%S%M%HT%d%b%yT")}_{info["run_id"]}__PLOT'
    else:
        sig = f'{info["Attack"]}__{info["clf"]}__{info["Start time"].strftime("T%S%M%HT%d%b%yT")}_{info["run_id"]}'
    info['Sig'] = sig
    info['Outpath'] = os.path.join(root_path, str(info['Sig']))

if __name__ == '__main__':
    func(info)
