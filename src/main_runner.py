import datetime
import os
import sys
import csv
import time
import tqdm
import shutil
from copy import deepcopy

from tools.Logger import Logger
import tools.paths as paths
import tools.config as config
import tools.params as params

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

import sklearn.metrics as metrics

from multiprocessing.pool import Pool
import tqdm

from sklearn.datasets.samples_generator import make_blobs

define_functions = True
if define_functions:
    def get_prob_of_red(pckg):
        global_idx = pckg['global_idx']
        paths = pckg['paths']
        config = pckg['config']
        params = pckg['params']

        logger = pckg['logger']
        item_idx = pckg['idx']
        row = pckg['row']
        cxdf = pckg['attack_df']
        data_cols = pckg['data_cols']
        adv_pos = pckg['adv_pos']

        label_to_reduce = pckg['label_to_reduce']
        clf_tag = pckg['clf_tag']
        clf = params['clf'][1]

        clf.fit(cxdf[data_cols], cxdf['label'])
        point = adv_pos[data_cols]
        prob_0, prob_1 = clf.predict_proba([point])[0]
        ret_pckg = dict()
        ret_pckg['idx'] = item_idx
        ret_pckg['prob_of_red'] = prob_1
        ret_pckg['prob_of_blue'] = prob_0
        return ret_pckg


    def simulate_creature(pckg):
        logger = pckg['logger']
        global_idx = pckg['global_idx']
        paths = pckg['paths']
        config = pckg['config']
        params = pckg['params']

        idx = pckg['idx']
        data_cols = pckg['data_cols']

        clf_tag = pckg['clf_tag']
        clf = params['clf'][1]
        cxdf = pckg['attack_df']
        creature = pckg['creature']

        adv_pos = pckg['adv_pos']

        # Attack
        xdf = cxdf.drop(creature.index)

        # Measure
        clf.fit(xdf[data_cols], xdf['label'])
        point = adv_pos[data_cols]
        prob_0, prob_1 = clf.predict_proba([point])[0]

        # Return
        ret_pckg = dict()
        ret_pckg['idx'] = idx
        ret_pckg['prob_of_red'] = prob_1
        ret_pckg['prob_of_blue'] = prob_0
        return ret_pckg


    def eval_clf(clf, X, y):
        y_pred = clf.predict(X)
        y_true = list(y)
        ret = metrics.accuracy_score(y_true, y_pred)

        return ret


    del define_functions


def main_func(info):
    code_init = True
    if code_init:
        define_env = True
        if define_env:
            global_idx = info['idx']
            paths = info['paths']
            config = info['config']
            params = info['params']

        # Old data removal
        clear_paths = True
        if clear_paths:
            paths_to_remove = dict()
            paths_to_remove['results'] = paths['result_path']
            for k, dir_path in paths_to_remove.iteritems():
                if os.path.exists(dir_path):
                    all_items_to_remove = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
                    for item_to_remove in all_items_to_remove:
                        if os.path.exists(item_to_remove) and not os.path.isdir(item_to_remove):
                            os.remove(item_to_remove)
                        else:
                            shutil.rmtree(item_to_remove)

            del k, dir_path
            del clear_paths, paths_to_remove

        # path verification
        verify_paths_exist = True
        if verify_paths_exist:
            paths_dict = dict()
            paths_dict['Root'] = paths['work_path']
            paths_dict['src'] = paths['src_path']
            paths_dict['results'] = paths['result_path']
            for k, folder_path in paths_dict.iteritems():
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

            del k, folder_path
            del verify_paths_exist, paths_dict

        # Define Logger
        define_logger = True
        if define_logger:
            # Output: logger
            logger = Logger(show_module=True)
            logger.initThread(paths['result_path'] + '\\report.txt')
            logger.log_print("START OF CODE")
            logger.log_print()

            del define_logger

        # Set random seed
        set_random_seed = True
        if set_random_seed:
            if config['random_seed'] is None:
                random_seed = np.random.randint(1, sys.maxint)
            else:
                random_seed = config['random_seed']
            np.random.seed(random_seed)
            logger.log_print("Random seed in use: {}".format(random_seed))
            logger.log_print()

            del set_random_seed, random_seed

        del code_init

    data_generation = True
    if data_generation:
        generate_synthetic_data = True
        if generate_synthetic_data:
            '''
            Output: 
            df: df with points and labels
            colors: dictionary with the color of each label
            data_cols
            '''

            logger.log_print("Creating data in <{}> format".format(params['type_of_input']['tag']))
            colors = {0: 'blue', 1: 'red', 2: 'black'}
            markers = {0: 'o', 1: 'o', 2: 'X'}
            if params['type_of_input']['tag'] == '2_blobs':
                dimensions = params['dimensions']
                clusters = params['clusters']
                samples = params['samples']
                test_ratio = params['test_size']
                clusters_std = params['type_of_input']['cluster_std']
                centers_pos = params['type_of_input']['centers']
                data_cols = ['P{}'.format(idx + 1) for idx in range(dimensions)]
                df = pd.DataFrame(columns=data_cols + ['label'], dtype='float32')
                test_df = pd.DataFrame(columns=data_cols + ['label'], dtype='float32')
                for group_idx in range(clusters):
                    xdf = pd.DataFrame(columns=['P{}'.format(idx + 1) for idx in range(dimensions)])
                    txdf = pd.DataFrame(columns=['P{}'.format(idx + 1) for idx in range(dimensions)])

                    X, y = make_blobs(n_samples=samples, n_features=dimensions,
                                      centers=1, cluster_std=clusters_std,
                                      center_box=centers_pos[group_idx],
                                      )
                    tX, ty = make_blobs(n_samples=int(test_ratio * samples), n_features=dimensions,
                                        centers=1, cluster_std=clusters_std,
                                        center_box=centers_pos[group_idx],
                                        )

                    for col in range(dimensions):
                        xdf['P{}'.format(col + 1)] = X[:, col]
                        txdf['P{}'.format(col + 1)] = tX[:, col]
                    xdf['label'] = group_idx
                    txdf['label'] = group_idx

                    df = df.append(xdf)
                    test_df = test_df.append(txdf)

                df = df.sample(frac=1).reset_index(drop=True)
                test_df = test_df.sample(frac=1).reset_index(drop=True)
                del col, idx, dimensions, clusters, clusters_std, centers_pos, samples, X, y
                del tX, ty
            if params['type_of_input']['tag'] == 'poly':
                dimensions = params['dimensions']
                clusters = params['clusters']
                samples = params['samples']
                crange = params['type_of_input']['range']
                points = np.random.uniform(low=crange[0], high=crange[1], size=(samples, dimensions))
                data_cols = ['P{}'.format(idx + 1) for idx in range(dimensions)]
                df = pd.DataFrame(columns=data_cols + ['label'], dtype='float32')
                for idx, col in enumerate(data_cols):
                    df[col] = points[:, idx]

                a, b, c, d = deepcopy(params['type_of_input']['a'], params['type_of_input']['b'],
                                      params['type_of_input']['c'],
                                      params['type_of_input']['d'], )

                def func(point):
                    ret = point['P2']
                    ret -= a
                    ret -= b * (point['P1'] ** 1)
                    ret -= c * (point['P1'] ** 2)
                    ret -= d * (point['P1'] ** 3)
                    return ret > 0

                df['label'] = df.apply(func, axis=1)

                del dimensions, clusters, samples, crange, points, idx, col
            del generate_synthetic_data

        del data_generation

    base_clf_measure = True
    if base_clf_measure:
        train_clf = True
        if train_clf:
            clf_tag, clf = params['clf']
            logger.log_print("fitting clf<{}> on data".format(clf_tag))
            clf.fit(df[data_cols], df['label'])

            res = eval_clf(clf, test_df[data_cols], test_df['label'])
            del train_clf

        del base_clf_measure

    create_adv_input = True
    if create_adv_input:
        # Output: adv_pos
        gen_adversarial = True
        if gen_adversarial:
            adv_pos = None
            tactic = params['adversarial_tactic']
            if tactic[0] == 'std_ratio':
                clusters = params['clusters']
                attack_info = tactic[1]

                # Get centroids
                centroids = pd.DataFrame()
                for label in set(df['label']):
                    xdf = df[df['label'] == label]
                    centeroid = xdf.mean()
                    centeroid.name = int(label)
                    centroids = centroids.append(centeroid)

                del xdf, centeroid

                # Get space of possible locations
                adversary_pos_center = pd.Series(index=data_cols).fillna(0)
                weight_sum = sum(attack_info['DISTANCES'])
                for idx, row in centroids.iterrows():
                    drow = row[data_cols]
                    dweight = attack_info['DISTANCES'][idx]
                    adversary_pos_center += (drow * (float(dweight) / weight_sum))

                del drow, dweight, idx, row, weight_sum

                # Get slope of line
                m = centroids[data_cols].iloc[1] - centroids[data_cols].iloc[0]
                m = m['P2'] / m['P1']
                adversary_pos_center['SLOPE'] = - 1.0 / m
                adversary_pos_center['OFFSET'] = adversary_pos_center['P2'] - (
                        adversary_pos_center['P2'] * adversary_pos_center['SLOPE'])

                del m
                # Get min-max bounds
                for p in data_cols:
                    adversary_pos_center['{}_min'.format(p)] = df[p].min()
                    adversary_pos_center['{}_max'.format(p)] = df[p].max()

                # Generate advesaries
                num_of_adversaries = 1
                for adversary in range(num_of_adversaries):
                    adv_pos = pd.Series(index=data_cols + ['label'])
                    adv_pos['label'] = 2
                    while True:

                        adv_pos['P1'] = np.random.uniform(low=adversary_pos_center['P1_min'] + 1.5,
                                                          high=adversary_pos_center['P1_max'] - 1.5)
                        adv_pos['P2'] = adv_pos['P1'] * adversary_pos_center['SLOPE'] + adversary_pos_center['OFFSET']

                        is_good = True
                        for p in data_cols:
                            is_good = is_good \
                                      and (
                                              (True and adversary_pos_center['{}_min'.format(p)] < adv_pos[p]) and
                                              adv_pos[
                                                  p] <
                                              adversary_pos_center['{}_max'.format(p)])
                        if is_good:
                            break
            del gen_adversarial

        del create_adv_input

    export_original_data = True
    if export_original_data:
        # Export raw data
        export_raw_data = True
        if export_raw_data:
            adv_pos.name = 'adv'
            df_xport = df.append(adv_pos, ignore_index=False)
            fname = 'data_with_adv_raw'
            outpath = os.path.join(paths['result_path'], '{}.csv'.format(fname))
            df_xport.to_csv(outpath)

            outpath = os.path.join(paths['result_path'], '{}.png'.format(fname))
            grouped = df_xport.groupby('label')
            fig, ax = plt.subplots()
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            for key, group in grouped:
                group.plot(ax=ax, kind='scatter', x='P1', y='P2', label=colors[key],
                           color=colors[key],
                           marker=markers[key])
            plt.savefig(outpath)
            plt.close()
            del export_raw_data
            del df_xport, outpath, grouped, fname

        # Export classefier with new point untouched
        plot_clf = True
        if plot_clf:
            logger.log_print("calculating contour")
            outpath = os.path.join(paths['result_path'], 'clf_BASE.png')
            X_test, y_test = df[data_cols], df['label']
            score = clf.score(X_test, y_test)
            title = '{} - base'.format(clf_tag)
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
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                out = ax.contourf(xx, yy, Z, **params)
                return out

            X0, X1, y = X_test['P1'], X_test['P2'], y_test
            xx, yy = make_meshgrid(X0, X1)
            plot_contours(ax, clf, xx, yy,
                          cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax.scatter(adv_pos['P1'], adv_pos['P2'], marker=markers[2])
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_xlabel('P1')
            ax.set_ylabel('P2')
            ax.set_title(title)
            plt.savefig(outpath)
            plt.close('all')
            del plot_clf
            del outpath
            del X0, X1, X_test, ax, fig, xx, y, yy, title, score

        del export_original_data

    # Attack and export data
    attack_and_export_data = True
    if attack_and_export_data:
        info_possible_budget = range(1, params['budget'] + 1)
        repetitions = params['repetitions']
        info_result_vector = pd.DataFrame()

        open_pool_if_needed = True
        if open_pool_if_needed:
            attack_info = params['attack_tactic'][1]
            if config.get('workload_handling', 'NULL') == 'parallel':
                logger.log_print("Opening pool")
                pool = Pool()
                logger.log_print("Pool opened.")
            del open_pool_if_needed

        get_original_prediction = True
        if get_original_prediction:
            budget = 0
            cols = list()
            cols += ['Budget']
            cols += ['Prob_Blue_rep_{}'.format(rep) for rep in range(repetitions)]
            cols += ['Prob_Blue']
            cols += ['Prob_Red_rep_{}'.format(rep) for rep in range(repetitions)]
            cols += ['Prob_Red']
            cols += ['Score_on_test_rep_{}'.format(rep) for rep in range(repetitions)]
            cols += ['Score_on_test']
            info_curr_row = pd.Series(name=budget, index=cols)
            info_curr_row['Budget'] = budget
            info_result_vector = info_result_vector.append(info_curr_row)

            for rep in range(repetitions):
                clf_tag, clf = params['clf']
                clf.fit(df[data_cols], df['label'])

                point = adv_pos[data_cols]
                prob_0, prob_1 = clf.predict_proba([point])[0]

                info_result_vector.loc[budget, 'Prob_Blue_rep_{}'.format(rep)] = prob_0
                info_result_vector.loc[budget, 'Prob_Red_rep_{}'.format(rep)] = prob_1
                info_result_vector.loc[budget, 'Score_on_test_rep_{}'.format(rep)] = eval_clf(clf, test_df[data_cols],
                                                                                              test_df['label'])

            info_result_vector.loc[budget, 'Prob_Blue'] = info_result_vector.filter(regex=("Prob_Blue_rep_[0-9]+")).loc[
                budget].mean()
            info_result_vector.loc[budget, 'Prob_Red'] = info_result_vector.filter(regex=("Prob_Red_rep_[0-9]+")).loc[
                budget].mean()
            info_result_vector.loc[budget, 'Score_on_test'] = \
                info_result_vector.filter(regex=("Score_on_test_rep_[0-9]+")).loc[
                    budget].mean()

            del get_original_prediction

        calculate_success_rate_for_each_budget = True
        if calculate_success_rate_for_each_budget:
            for budget in info_possible_budget:
                cols = list()
                cols += ['Budget']
                cols += ['Prob_Blue_rep_{}'.format(rep) for rep in range(repetitions)]
                cols += ['Prob_Blue']
                cols += ['Prob_Red_rep_{}'.format(rep) for rep in range(repetitions)]
                cols += ['Prob_Red']
                cols += ['Score_on_test_rep_{}'.format(rep) for rep in range(repetitions)]
                cols += ['Score_on_test']

                info_curr_row = pd.Series(name=budget, index=cols)
                info_curr_row['Budget'] = budget
                info_result_vector = info_result_vector.append(info_curr_row)

                for rep in range(repetitions):
                    curr_sig = 'BUDGET={} of {} rep={}'.format(budget, info_possible_budget[-1], rep)
                    logger.switcheModule(curr_sig)

                    attack_data = True
                    if attack_data:
                        attack_df = df.copy(deep=True)
                        attack_info = params['attack_tactic'][1]
                        attack_sig = params['attack_tactic'][0]

                        if attack_sig == 'by_distance_from_adv':
                            # Compute distances, Output: dstdf
                            compute_distances = True
                            if compute_distances:
                                def get_dist(p1, p2, data_cols):
                                    ret = 0
                                    ret += (p1[data_cols] - p2[data_cols])
                                    ret = ret ** 2
                                    ret = ret.sum()
                                    ret = np.sqrt(ret)
                                    return ret

                                attack_df['dist_from_adv'] = df.apply(lambda row: get_dist(row, adv_pos, data_cols),
                                                                      axis=1)
                                attack_df['dist_from_blue'] = df.apply(
                                    lambda row: get_dist(row, centroids[centroids['label'] == 0].iloc[0], data_cols),
                                    axis=1)
                                attack_df['dist_from_red'] = df.apply(
                                    lambda row: get_dist(row, centroids[centroids['label'] == 1].iloc[0], data_cols),
                                    axis=1)
                                del compute_distances

                            remove_from_blue = True
                            if remove_from_blue:
                                blue_budget = int(budget * attack_info['budget_for_remove_blue'])
                                blue_dsdf = attack_df[attack_df['label'] == 0]
                                blue_dsdf = blue_dsdf.sort_values(by=['dist_from_adv'], ascending=False)
                                blue_dsdf = blue_dsdf.iloc[0:blue_budget]
                                blue_dsdf_index = blue_dsdf.index

                                attack_df = attack_df.drop(blue_dsdf_index)
                                del remove_from_blue
                                del blue_dsdf, blue_dsdf_index

                            # Removing from red
                            remove_from_red = True
                            if remove_from_red:
                                red_budget = int(budget * attack_info['budget_for_remove_red'])
                                red_dsdf = attack_df[attack_df['label'] == 1]
                                red_dsdf = red_dsdf.sort_values(by=['dist_from_adv'], ascending=True)
                                red_dsdf = red_dsdf.iloc[0:red_budget]
                                red_dsdf_index = red_dsdf.index

                                attack_df = attack_df.drop(red_dsdf_index)
                                del remove_from_red
                                del red_dsdf, red_dsdf_index

                        if attack_sig == 'greedy_search':
                            k = budget
                            workload = config.get('workload_handling', 'NULL')

                            for kidx in range(k):
                                # Generate input
                                input_vector = list()
                                for item_idx, row in attack_df.iterrows():
                                    pckg = dict()
                                    pckg['idx'] = item_idx
                                    pckg['row'] = row
                                    pckg['attack_df'] = attack_df.drop([item_idx])
                                    pckg['clf_tag'] = clf_tag
                                    pckg['adv_pos'] = adv_pos
                                    pckg['data_cols'] = data_cols
                                    pckg['label_to_reduce'] = 1
                                    pckg['logger'] = None
                                    pckg['params'] = params
                                    pckg['config'] = config
                                    pckg['paths'] = paths
                                    pckg['global_idx'] = global_idx

                                    input_vector.append(pckg)

                                calculate_prob_of_each_removal = True
                                if calculate_prob_of_each_removal:
                                    result_vector = list()
                                    tqdm_msg = '[{}]\tRound ({:>3}/{:>3})'.format(workload, int(kidx + 1), int(k))

                                    if workload == 'concurrent':
                                        for item_id in tqdm.tqdm(range(len(input_vector)), desc=tqdm_msg, ascii=True):
                                            ret = get_prob_of_red(input_vector[item_id])
                                            result_vector.append(ret)
                                    if workload == 'parallel':
                                        results = pool.imap(get_prob_of_red, input_vector)
                                        for item_id in tqdm.tqdm(range(len(input_vector)), desc=tqdm_msg, ascii=True):
                                            ret = results.next()
                                            result_vector.append(ret)

                                    del calculate_prob_of_each_removal
                                    del ret, item_id, tqdm_msg

                                choose_point_to_remove = True
                                if choose_point_to_remove:
                                    prob_of_red = pd.Series(index=df.index)
                                    for res in result_vector:
                                        prob_of_red[res['idx']] = res['prob_of_red']

                                    best_idx_to_remove = prob_of_red.idxmin()
                                    attack_df = attack_df.drop([best_idx_to_remove])

                                    del choose_point_to_remove
                                    del res, best_idx_to_remove, prob_of_red

                            del k

                        if attack_sig == 'Genetic':
                            k = budget
                            workload = config.get('workload_handling', 'NULL')

                            generations = attack_info['generations']
                            mutation_rate = 1.0 / k
                            offsprings = attack_info['offsprings']
                            parents = attack_info['parents']
                            result_summary = pd.DataFrame(columns=['Gen', 'Best score'])

                            # Generate original parents
                            generate_parents = True
                            if generate_parents:
                                parents_dict = dict()
                                for offspring in range(offsprings):
                                    parent = attack_df.sample(k)
                                    parents_dict[offspring] = parent
                                del generate_parents

                            # iterate generation
                            evolve = True
                            if evolve:
                                for gen in range(generations):

                                    # Generate inputs
                                    input_vector = list()
                                    for creature_idx, creature in parents_dict.iteritems():
                                        pckg = dict()
                                        pckg['idx'] = creature_idx
                                        pckg['creature'] = creature
                                        pckg['attack_df'] = attack_df
                                        pckg['clf_tag'] = clf_tag
                                        pckg['adv_pos'] = adv_pos
                                        pckg['data_cols'] = data_cols
                                        pckg['logger'] = None
                                        pckg['params'] = params
                                        pckg['config'] = config
                                        pckg['paths'] = paths
                                        pckg['global_idx'] = global_idx
                                        input_vector.append(pckg)

                                    # simulate attacks
                                    simulate_attacks = True
                                    if simulate_attacks:
                                        result_vector = list()
                                        tqdm_msg = '[{}]\tGeneration ({:>3}/{:>3})'.format(workload, int(gen + 1),
                                                                                           int(generations))

                                        if workload == 'concurrent':
                                            for item_id in tqdm.tqdm(range(len(input_vector)), desc=tqdm_msg,
                                                                     ascii=True):
                                                ret = simulate_creature(input_vector[item_id])
                                                result_vector.append(ret)
                                        if workload == 'parallel':
                                            results = pool.imap(simulate_creature, input_vector)
                                            for item_id in tqdm.tqdm(range(len(input_vector)), desc=tqdm_msg,
                                                                     ascii=True):
                                                ret = results.next()
                                                result_vector.append(ret)

                                        del simulate_attacks
                                        del ret, item_id, tqdm_msg

                                    # Create new generation
                                    create_new_gen = True
                                    if create_new_gen:
                                        gendf = pd.DataFrame(result_vector)
                                        gendf = gendf.sort_values(by=['prob_of_blue'], ascending=False)
                                        gendf = gendf.iloc[:parents]

                                        # Get winner
                                        winner_creature = parents_dict[int(gendf.iloc[0]['idx'])]
                                        result_summary = result_summary.append(
                                            {'Gen': gen + 1, 'Best score': gendf.iloc[0]['prob_of_blue']},
                                            ignore_index=True)

                                        new_parents_dict = dict()
                                        for parent in range(parents):
                                            new_parents_dict[parent] = parents_dict[int(gendf.iloc[parent]['idx'])]

                                        genepool = pd.DataFrame()
                                        for parent_idx, parent in new_parents_dict.iteritems():
                                            genepool = genepool.append(parent)
                                        genepool = genepool[~genepool.index.duplicated(keep='first')]

                                        for offspring_idx in range(parents, offsprings):
                                            offspring = genepool.sample(k)

                                            # mutate
                                            mutate_offspring = True
                                            if mutate_offspring:
                                                offspring['REPLACE'] = [
                                                    np.random.choice([True, False],
                                                                     p=[mutation_rate, 1 - mutation_rate])
                                                    for _ in range(k)]

                                                items_dropped = sum(offspring['REPLACE'])
                                                offspring = offspring[offspring['REPLACE'] != True]

                                                offspring = offspring.drop(['REPLACE'], axis=1)
                                                mutation_pool = attack_df.drop(attack_df.index[offspring.index])
                                                offspring = offspring.append(mutation_pool.sample(items_dropped),
                                                                             sort=False)

                                                del mutate_offspring, items_dropped, mutation_pool

                                            new_parents_dict[offspring_idx] = offspring

                                        for kt in parents_dict.keys():
                                            del parents_dict[kt]
                                        parents_dict = new_parents_dict

                                        del create_new_gen, genepool

                                del evolve

                            # attack with winner
                            attack_with_winner = True
                            if attack_with_winner:
                                attack_df = attack_df.drop(winner_creature.index)

                                del attack_with_winner

                            export_evolution_data = True
                            if export_evolution_data:
                                export_title = 'Evolution_{}'.format(curr_sig)
                                export_path_csv = os.path.join(paths['result_path'], '{}.csv'.format(export_title))
                                export_path_png = os.path.join(paths['result_path'], '{}.png'.format(export_title))
                                result_summary.to_csv(export_path_csv, index=False)

                                plot = result_summary.plot.line('Gen', 'Best score')
                                plot.set_ybound(0, 1)
                                fig = plot.get_figure()
                                fig.savefig(export_path_png)
                                plt.close('all')
                                del export_path_csv, export_path_png
                                del export_evolution_data, result_summary

                        del attack_data, attack_sig, attack_info

                    train_clf = True
                    if train_clf:
                        clf_tag, clf = params['clf']
                        logger.log_print("fitting clf<{}> on data".format(clf_tag))
                        clf.fit(attack_df[data_cols], attack_df['label'])
                        clf_score = eval_clf(clf, test_df[data_cols], test_df['label'])

                        point = adv_pos[data_cols]
                        prob_0, prob_1 = clf.predict_proba([point])[0]
                        info_result_vector.loc[budget, 'Prob_Blue_rep_{}'.format(rep)] = prob_0
                        info_result_vector.loc[budget, 'Prob_Red_rep_{}'.format(rep)] = prob_1
                        info_result_vector.loc[budget, 'Score_on_test_rep_{}'.format(rep)] = clf_score

                        del train_clf
                        del point, prob_0, prob_1

                    # Export raw data
                    export_raw_data = True
                    if export_raw_data:
                        df_xport = attack_df.append(adv_pos, ignore_index=True)
                        outpath = os.path.join(paths['result_path'], 'data_with_adv_ATTACKED_{}.png'.format(curr_sig))

                        grouped = df_xport.groupby('label')
                        fig, ax = plt.subplots()
                        ax.set_xlim([-10, 10])
                        ax.set_ylim([-10, 10])
                        for key, group in grouped:
                            group.plot(ax=ax, kind='scatter', x='P1', y='P2', label=colors[key],
                                       color=colors[key],
                                       marker=markers[key])
                        plt.savefig(outpath)
                        plt.close('all')
                        del export_raw_data
                        del df_xport, outpath, grouped

                    # Export classefier with new point untouched
                    plot_clf = True
                    if plot_clf:
                        logger.log_print("calculating contour")
                        outpath = os.path.join(paths['result_path'], 'clf_ATTACKED_{}.png'.format(curr_sig))
                        X_test, y_test = attack_df[data_cols], attack_df['label']
                        score = clf.score(X_test, y_test)
                        title = '{} - ATTACKED ({}) - [{}]'.format(clf_tag, params['attack_tactic'][0], curr_sig)
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
                            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                            Z = Z.reshape(xx.shape)
                            out = ax.contourf(xx, yy, Z, **params)
                            return out

                        X0, X1, y = X_test['P1'], X_test['P2'], y_test
                        xx, yy = make_meshgrid(X0, X1)
                        plot_contours(ax, clf, xx, yy,
                                      cmap=plt.cm.coolwarm, alpha=0.8)
                        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
                        ax.scatter(adv_pos['P1'], adv_pos['P2'], marker=markers[2])
                        ax.set_xlim([-10, 10])
                        ax.set_ylim([-10, 10])
                        ax.set_xlabel('P1')
                        ax.set_ylabel('P2')
                        ax.set_title(title)
                        plt.savefig(outpath)
                        plt.close('all')

                        del plot_clf
                        del outpath
                        del X0, X1, X_test, ax, fig, xx, y, yy, title, score

                calculate_results_for_current_budget = True
                if calculate_results_for_current_budget:
                    info_result_vector.loc[budget, 'Prob_Blue'] = \
                        info_result_vector.filter(regex=("Prob_Blue_rep_[0-9]+")).loc[
                            budget].mean()
                    info_result_vector.loc[budget, 'Prob_Red'] = \
                        info_result_vector.filter(regex=("Prob_Red_rep_[0-9]+")).loc[
                            budget].mean()

                    info_result_vector.loc[budget, 'Score_on_test'] = \
                        info_result_vector.filter(regex=("Score_on_test_rep_[0-9]+")).loc[
                            budget].mean()

                    del calculate_results_for_current_budget

        close_pool_if_needed = True
        if close_pool_if_needed:
            attack_info = params['attack_tactic'][1]
            if config.get('workload_handling', 'NULL') == 'parallel':
                pool.close()
            del close_pool_if_needed

        logger.switcheModuleToDefault()

        del attack_and_export_data

    # export attacked data
    export_attacked_data = True
    if export_attacked_data:
        outname = 'Success over budget'
        outpath = os.path.join(paths['result_path'], '{}.csv'.format(outname))
        info_result_vector.to_csv(outpath, index=False)

        outpath = os.path.join(paths['result_path'], '{}.png'.format(outname))
        fig, ax = plt.subplots(1)
        ax.plot(info_result_vector['Budget'], info_result_vector['Prob_Blue'], marker='o', color='blue',
                label='Prob of adv. to blue')
        ax.plot(info_result_vector['Budget'], info_result_vector['Score_on_test'], marker='o', color='red',
                label='test group success')
        ax.legend()
        ax.axhline(y=0.5, color='yellow')
        ax.set_xticks(np.arange(0, info_possible_budget[-1] + 1, step=1))
        ax.set_ybound(0, 1)
        ax.set_xlabel("Budget")

        fig.savefig(outpath)
        plt.close('all')

        del outname, outpath, plot, fig
        del export_attacked_data

    code_fin = True
    if code_fin:
        logger.log_print()
        logger.system_check("End of code system check")
        logger.log_print("END OF CODE.")
        logger.log_close()


# Define inputs
if __name__ == '__main__':

    define_inputs = True
    if define_inputs:
        info_base = dict()
        info_base['idx'] = None
        info_base['paths'] = paths.itemized
        info_base['params'] = params.itemized
        info_base['config'] = config.itemized

        # Set random seed
        set_random_seed = True
        if set_random_seed:
            user_seed = config.random_seed
            if user_seed is None:
                random_seed = np.random.randint(1, sys.maxint)
            else:
                random_seed = user_seed
            np.random.seed(random_seed)

            del set_random_seed, random_seed, user_seed

        info_vector = list()
        for idx in range(20):
            cinfo = deepcopy(info_base)
            cinfo['idx'] = idx
            cinfo['paths']['result_path'] = (cinfo['paths']['result_path']).replace('@', str(idx))
            cinfo['config']['random_seed'] = np.random.randint(1, sys.maxint)
            info_vector.append(cinfo)

        del define_inputs

# Run code
if __name__ == '__main__':
    for info in info_vector:
        main_func(info)
