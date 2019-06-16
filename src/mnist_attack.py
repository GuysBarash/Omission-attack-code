import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn.svm import SVC
import tqdm

import os
from multiprocessing.pool import Pool
import shutil

from tools.Logger import Logger
from tools.paths import itemized as paths
from tools.params import itemized as params


def simulate_creature(pckg):
    logger = pckg['logger']
    global_idx = pckg['global_idx']
    params = pckg['params']

    parent_idx = pckg['parent_idx']
    data_cols = pckg['data_cols']
    from_label = pckg['from_label']
    to_label = pckg['to_label']

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

    from_label_idx = np.where(clf.classes_ == from_label)[0][0]
    to_label_idx = np.where(clf.classes_ == to_label)[0][0]
    probs = clf.predict_proba([point])[0]
    prob_origin, prob_target = probs[from_label_idx], probs[to_label_idx]

    # Return
    ret_pckg = dict()
    ret_pckg['parent_idx'] = parent_idx
    ret_pckg['prob_of_origin'] = prob_origin
    ret_pckg['prob_of_target'] = prob_target
    return ret_pckg


def plot_sample(img, path, title=None):
    mat = img.values.reshape((28, 28))
    imgplot = plt.imshow(mat)
    if title is not None:
        plt.title(title)
    plt.savefig(path)
    plt.close()


def visualize_by_label(df, label_col, outputpath, fname='NoName', components=2):
    outputpath = os.path.join(outputpath, '{}.png'.format(fname))

    data = df.drop(columns=[label_col])
    labels = df[label_col]

    pca = PCA(n_components=components)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, labels], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = labels.unique()
    tags = len(targets)
    colors = cm.rainbow(np.linspace(0, 1, tags))

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf[label_col] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   c=color,
                   s=50,
                   )
    ax.legend(targets)
    ax.grid()
    plt.savefig(outputpath)
    plt.close()


def attack(df, adv_p, attack_info, logger):
    budget = attack_info['budget']
    info_possible_budget = [budget]
    from_label = attack_info['from_label']
    to_label = attack_info['to_label']
    label_col = attack_info['label_col']
    data_cols = attack_info['data_cols']
    workload = attack_info['workload']
    repetitions = attack_info['repetitions']
    attack_tactic = attack_info['attack_tactic']
    global_idx = attack_info['global_idx']
    store_results = attack_info['store_results']
    stop_on_success = attack_info['stop_on_success']
    results_path = attack_info['results_path']

    info_result_vector = pd.DataFrame()

    open_pool_if_needed = True
    if open_pool_if_needed:
        if workload == 'parallel':
            logger.log_print("Opening pool")
            pool = Pool()
            logger.log_print("Pool opened.")
        del open_pool_if_needed

    get_original_prediction = True
    if get_original_prediction:
        budget = 0
        cols = list()
        cols += ['Budget']
        cols += ['Prob_origin_rep_{}'.format(rep) for rep in range(repetitions)]
        cols += ['Prob_origin']
        cols += ['Prob_target_rep_{}'.format(rep) for rep in range(repetitions)]
        cols += ['Prob_target']
        cols += ['WIN']
        cols += ['Score_on_test_rep_{}'.format(rep) for rep in range(repetitions)]
        cols += ['Score_on_test']
        info_curr_row = pd.Series(name=budget, index=cols)
        info_curr_row['Budget'] = budget
        info_result_vector = info_result_vector.append(info_curr_row)

        for rep in range(repetitions):
            clf_tag, clf = params['clf']
            clf.fit(df[data_cols], df['label'])

            point = adv_p[data_cols]
            from_label_idx = np.where(clf.classes_ == from_label)[0][0]
            to_label_idx = np.where(clf.classes_ == to_label)[0][0]
            probs = clf.predict_proba([point])[0]
            prob_origin, prob_target = probs[from_label_idx], probs[to_label_idx]

            info_result_vector.loc[budget, 'Prob_origin_rep_{}'.format(rep)] = prob_origin
            info_result_vector.loc[budget, 'Prob_target_rep_{}'.format(rep)] = prob_target
            info_result_vector.loc[budget, 'Score_on_test_rep_{}'.format(rep)] = 0

        info_result_vector.loc[budget, 'Prob_origin'] = info_result_vector.filter(regex=("Prob_origin_rep_[0-9]+")).loc[
            budget].mean()
        info_result_vector.loc[budget, 'Prob_target'] = info_result_vector.filter(regex=("Prob_target_rep_[0-9]+")).loc[
            budget].mean()

        info_result_vector.loc[budget, 'WIN'] = info_result_vector.loc[budget, 'Prob_target'] >= 0.5
        info_result_vector.loc[budget, 'Score_on_test'] = \
            info_result_vector.filter(regex=("Score_on_test_rep_[0-9]+")).loc[
                budget].mean()

        del get_original_prediction

    calculate_success_rate_for_each_budget = True
    if calculate_success_rate_for_each_budget:
        for budget in info_possible_budget:
            cols = list()
            cols += ['Budget']
            cols += ['Prob_origin_rep_{}'.format(rep) for rep in range(repetitions)]
            cols += ['Prob_origin']
            cols += ['Prob_target_rep_{}'.format(rep) for rep in range(repetitions)]
            cols += ['Prob_target']
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
                    attack_info = attack_tactic[1]
                    attack_sig = attack_tactic[0]

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
                        greedy_restarts = attack_info['iteration_budget']
                        for current_restart_idx in range(1):

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
                                        for item_id in tqdm.tqdm(range(len(input_vector)), desc=tqdm_msg,
                                                                 ascii=True):
                                            ret = get_prob_of_red(input_vector[item_id])
                                            result_vector.append(ret)
                                    if workload == 'parallel':
                                        results = pool.imap(get_prob_of_red, input_vector)
                                        for item_id in tqdm.tqdm(range(len(input_vector)), desc=tqdm_msg,
                                                                 ascii=True):
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

                        generations = attack_info['generations']
                        mutation_rate = 1.0 / k
                        offsprings = attack_info['offsprings']
                        parents = attack_info['parents']
                        per_round_winner = pd.DataFrame(
                            columns=['Gen', 'creature_idx', 'prob_of_origin',
                                     'prob_of_target', 'change'] + [
                                        's_{}'.format(tk) for tk in range(k)])
                        result_summary = pd.DataFrame(columns=['Gen', 'Best score'])
                        results_memory = pd.DataFrame(columns=['parent_idx', 'prob_of_origin', 'prob_of_target'])

                        # Generate original parents
                        generate_parents = True
                        if generate_parents:
                            parents_dict = dict()
                            for offspring in range(offsprings - 1):
                                parent = attack_df.sample(k).sort_index()
                                parents_dict[offspring] = parent

                            # one of the parents is always KNN
                            get_KNN_winner = True
                            if get_KNN_winner:
                                if attack_info['KNN_parent']:
                                    def get_dist_func(p1, p2, data_cols):
                                        ret = 0
                                        ret += (p1[data_cols] - p2[data_cols])
                                        ret = ret ** 2
                                        ret = ret.sum()
                                        ret = np.sqrt(ret)
                                        return ret

                                    tparent = attack_df.copy(deep=True)
                                    tparent['dist_from_adv'] = tparent.apply(
                                        lambda row: get_dist_func(row, adv_p, data_cols),
                                        axis=1)

                                    tparent = tparent[tparent['label'] == 1.0]
                                    tparent = tparent.sort_values(by=['dist_from_adv'], ascending=True)
                                    tparent = tparent.iloc[0:k]
                                    tparent = tparent.drop(columns=['dist_from_adv'])
                                    tparent = tparent.sort_index()
                                    parents_dict[offsprings - 1] = tparent

                                    del tparent
                                    del get_KNN_winner
                                else:
                                    tparent = attack_df.sample(k).sort_index()
                                    parents_dict[offsprings - 1] = tparent

                            del generate_parents

                        # iterate generation
                        evolve = True
                        if evolve:
                            curr_score = 0
                            for gen in range(generations):
                                if curr_score > 0.9999:
                                    break

                                # Generate inputs
                                input_vector = list()
                                result_vector = list()
                                for creature_idx, creature in parents_dict.iteritems():
                                    pckg = dict()
                                    pckg['parent_idx'] = creature_idx
                                    pckg['creature'] = creature
                                    pckg['attack_df'] = attack_df
                                    pckg['clf_tag'] = clf_tag
                                    pckg['adv_pos'] = adv_p
                                    pckg['data_cols'] = data_cols
                                    pckg['logger'] = None
                                    pckg['params'] = params
                                    pckg['paths'] = paths
                                    pckg['to_label'] = to_label
                                    pckg['from_label'] = from_label
                                    pckg['global_idx'] = global_idx
                                    pckg['HASH'] = hash(bytes(creature.index))

                                    if pckg['HASH'] in results_memory.index:
                                        ret = dict(results_memory.loc[pckg['HASH']])
                                        ret['parent_idx'] = creature_idx
                                        result_vector.append(ret)

                                    else:
                                        input_vector.append(pckg)

                                # simulate attacks
                                simulate_attacks = True
                                if simulate_attacks:

                                    tqdm_msg = '[{}][Score: {:>4.3f}][Budget: {:>4.3f}]\tGeneration ({:>3}/{:>3})'.format(
                                        workload,
                                        curr_score,
                                        k,
                                        int(gen + 1),
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
                                    del tqdm_msg

                                # Store results to memory
                                if store_results:
                                    for res in result_vector:
                                        creature_hash = hash(bytes(parents_dict[res['parent_idx']].index))
                                        results_memory.loc[creature_hash] = res

                                # Create new generation
                                create_new_gen = True
                                if create_new_gen:
                                    gendf = pd.DataFrame(result_vector)
                                    gendf = gendf.sort_values(by=['prob_of_target'], ascending=False)

                                    if curr_score > gendf.iloc[0]['prob_of_target']:
                                        print "BUG!!!"

                                    gendf = gendf.iloc[:parents]

                                    # Get winner
                                    winner_creature_idx = int(gendf.iloc[0]['parent_idx'])
                                    winner_creature = parents_dict[winner_creature_idx]
                                    result_summary = result_summary.append(
                                        {'Gen': gen + 1, 'Best score': gendf.iloc[0]['prob_of_target']},
                                        ignore_index=True)

                                    old_score = curr_score
                                    curr_score = gendf.iloc[0]['prob_of_target']
                                    winner_sr = pd.Series(index=per_round_winner.columns)
                                    winner_sr['Gen'] = gen
                                    winner_sr['creature_idx'] = winner_creature_idx
                                    winner_sr['prob_of_origin'] = gendf.iloc[0]['prob_of_origin']
                                    winner_sr['prob_of_target'] = gendf.iloc[0]['prob_of_target']
                                    winner_sr['change'] = curr_score - old_score
                                    for kt in range(k):
                                        winner_sr['s_{}'.format(kt)] = winner_creature.index[kt]
                                    per_round_winner = per_round_winner.append(winner_sr, ignore_index=True)

                                    new_parents_dict = dict()
                                    for parent in range(parents):
                                        original_parent_idx = int(gendf.iloc[parent]['parent_idx'])
                                        original_parent_item = parents_dict[original_parent_idx]
                                        new_parents_dict[parent] = original_parent_item

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
                                            mutation_pool = attack_df.drop(offspring.index)
                                            offspring = offspring.append(mutation_pool.sample(items_dropped),
                                                                         sort=False)

                                            del mutate_offspring, items_dropped, mutation_pool

                                        offspring = offspring.sort_index()
                                        new_parents_dict[offspring_idx] = offspring

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
                            export_path_csv = os.path.join(results_path, '{}.csv'.format(export_title))
                            export_path_png = os.path.join(results_path, '{}.png'.format(export_title))
                            per_gen_winner_path = os.path.join(results_path, 'per_round_winner.csv')
                            result_summary.to_csv(export_path_csv, index=False)
                            per_round_winner.to_csv(per_gen_winner_path, index=False)

                            plot = result_summary.plot.line('Gen', 'Best score')
                            plot.set_ybound(0, 1)
                            fig = plot.get_figure()
                            fig.savefig(export_path_png)
                            plt.close('all')
                            del export_path_csv, export_path_png, per_gen_winner_path, plot
                            del export_evolution_data, result_summary

                    del attack_data, attack_sig, attack_info

                train_clf = True
                if train_clf:
                    clf_tag, clf = params['clf']
                    logger.log_print("fitting clf<{}> on data".format(clf_tag))
                    clf.fit(attack_df[data_cols], attack_df['label'])
                    clf_score = 0  # eval_clf(clf, test_df[data_cols], test_df['label'])

                    point = adv_p[data_cols]
                    from_label_idx = np.where(clf.classes_ == from_label)[0][0]
                    to_label_idx = np.where(clf.classes_ == to_label)[0][0]
                    probs = clf.predict_proba([point])[0]
                    prob_origin, prob_target = probs[from_label_idx], probs[to_label_idx]

                    info_result_vector.loc[budget, 'Prob_origin_rep_{}'.format(rep)] = prob_origin
                    info_result_vector.loc[budget, 'Prob_target_rep_{}'.format(rep)] = prob_target
                    info_result_vector.loc[budget, 'Score_on_test_rep_{}'.format(rep)] = 0

                    del train_clf
                    del point, prob_target, prob_origin

                plot_pca_of_world = False
                if plot_pca_of_world:

                    # Export raw data
                    export_raw_data = False
                    if export_raw_data:
                        df_xport = attack_df.append(adv_p, ignore_index=True)
                        outpath = os.path.join(results_path, 'data_with_adv_ATTACKED_{}.png'.format(curr_sig))

                        grouped = df_xport.groupby('label')
                        fig, ax = plt.subplots()
                        ax.set_xlim([-10, 10])
                        ax.set_ylim([-10, 10])
                        colors = {0: 'blue', 1: 'red', 2: 'black'}
                        markers = {0: 'o', 1: 'o', 2: 'X'}
                        for key, group in grouped:
                            group.plot(ax=ax, kind='scatter', x='P1', y='P2', label=colors[key],
                                       color=colors[key],
                                       marker=markers[key])
                        plt.savefig(outpath)
                        plt.close('all')
                        del export_raw_data
                        del df_xport, outpath, grouped

                    # Export classefier with new point untouched
                    plot_clf = False
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

                        colors = {0: 'blue', 1: 'red', 2: 'black'}
                        markers = {0: 'o', 1: 'o', 2: 'X'}
                        X0, X1, y = X_test['P1'], X_test['P2'], y_test
                        xx, yy = make_meshgrid(X0, X1)
                        plot_contours(ax, clf, xx, yy,
                                      cmap=plt.cm.coolwarm, alpha=0.8)
                        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
                        ax.scatter(adv_p['P1'], adv_p['P2'], marker=markers[2])
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

                    del plot_pca_of_world

                plot_creature_samples = True
                if plot_creature_samples:
                    outpath = os.path.join(results_path, 'visualize')
                    if not os.path.exists(outpath):
                        os.makedirs(outpath)

                    adv_path = os.path.join(outpath, 'adverserial_point.png')
                    plot_sample(adv_p.drop(label_col), adv_path,
                                title='original class: {}, target_class: {}'.format(from_label, to_label))
                    for cidx, (sample_to_remove_idx, sample_to_remove_sr) in enumerate(winner_creature.iterrows()):
                        title = 'Sample {}  Label: {}'.format(sample_to_remove_idx, sample_to_remove_sr[label_col])
                        sample_path = os.path.join(outpath, '{}_sample_{}_label_{}.png'.format(
                            cidx, sample_to_remove_idx, sample_to_remove_sr[label_col]
                        ))
                        plot_sample(sample_to_remove_sr.drop(label_col), sample_path, title)

                    del plot_creature_samples

            calculate_results_for_current_budget = True
            if calculate_results_for_current_budget:
                aggregation = params['aggregation_function']
                aggreg_func_blue = aggregation['aggreg_func_blue']
                aggreg_func_red = aggregation['aggreg_func_red']
                info_result_vector.loc[budget, 'Prob_Blue'] = aggreg_func_blue(
                    info_result_vector.filter(regex="Prob_Blue_rep_[0-9]+").loc[
                        budget])
                info_result_vector.loc[budget, 'Prob_Red'] = aggreg_func_red(
                    info_result_vector.filter(regex="Prob_Red_rep_[0-9]+").loc[
                        budget])

                info_result_vector.loc[budget, 'WIN'] = info_result_vector.loc[budget, 'Prob_Blue'] >= 0.5
                current_round_victory = info_result_vector.loc[budget, 'WIN']

                info_result_vector.loc[budget, 'Score_on_test'] = \
                    info_result_vector.filter(regex="Score_on_test_rep_[0-9]+").loc[
                        budget].mean()

                del calculate_results_for_current_budget

            stop_condition_on_budget = True
            if stop_condition_on_budget:
                if current_round_victory:
                    logger.log_print("Victory was achieved with budget {}".format(budget))
                    outpath = os.path.join(paths['result_path'], 'A_VICTORY_AT_{}'.format(budget))
                    with open(outpath, 'w') as ffile:
                        ffile.write("DONE.")
                        ffile.close()

                    if stop_on_success:
                        break

                    del outpath, ffile

                del stop_condition_on_budget

    close_pool_if_needed = True
    if close_pool_if_needed:
        if workload == 'parallel':
            pool.close()
        del close_pool_if_needed


if __name__ == '__main__':
    # Old data removal
    clear_paths = True
    if clear_paths:
        paths_to_remove = dict()
        paths_to_remove['data'] = paths['data_path']
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
        paths_dict['data'] = paths['data_path']
        for k, folder_path in paths_dict.iteritems():
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        del k, folder_path
        del verify_paths_exist, paths_dict

    init_logger = True
    if init_logger:
        logger = Logger(show_module=True)
        logger.initThread(paths['data_path'] + '\\report.txt')
        logger.log_print("START OF CODE")
        logger.log_print()

        del init_logger

if __name__ == '__main__':
    open_df = True
    if open_df:
        label_col = 'label'
        csv_path = r"C:\school\thesis\omission\mnist\mnist_train.csv"
        df = pd.read_csv(csv_path)
        df = df.sample(frac=1)
        relevant_labels = [3, 4]
        df = df[df[label_col].isin(relevant_labels)]
        df = df.iloc[0:200]

        del open_df

    find_hardest_samples = True
    if find_hardest_samples:
        clf_name, clf = params['clf']
        clf.fit(df.drop([label_col], axis=1), df[label_col])

        res = clf.predict_proba(df.drop([label_col], axis=1))
        res = pd.DataFrame(res, index=df.index)
        res['diff'] = np.abs(res[0] - res[1])
        res = res.sort_values(by=['diff'])
        top_n = 5

        # Print top candidates
        top_idx = res.iloc[0:top_n].index
        for idx in top_idx:
            img = df.loc[idx].drop(label_col)
            label = df.loc[idx][label_col]
            path = os.path.join(paths['data_path'], 'img_{}.png'.format(idx))
            logger.log_print("Print suspect to {}".format(path))
            title = '[prob of {l1}: {p1:>.2f}][prob of {l2}: {p2:>.2f}]'.format(
                l1=relevant_labels[1], l2=relevant_labels[0],
                p1=res.loc[idx][0], p2=res.loc[idx][1],
            )
            plot_sample(img, path, title=title)

        adv_idx = top_idx[0]
        adv = df.loc[adv_idx]
        xdf = df.drop(adv_idx)
        xdf.reset_index(drop=True)

        del find_hardest_samples
        del res

    commit_attack = True
    if commit_attack:
        attack_info = dict()
        attack_info['budget'] = int(np.ceil(np.sqrt(xdf.shape[0])))
        attack_info['from_label'] = adv[label_col]
        attack_info['to_label'] = [t for t in xdf[label_col].unique() if t != adv[label_col]][0]
        attack_info['workload'] = 'parallel'  # 'parallel' , 'concurrent'
        attack_info['repetitions'] = 1
        attack_info['label_col'] = label_col
        attack_info['data_cols'] = [t for t in adv.index if t != label_col]
        attack_info['attack_tactic'] = params['attack_tactics']['Genetic']
        attack_info['global_idx'] = 0
        attack_info['store_results'] = False
        attack_info['stop_on_success'] = True
        attack_info['results_path'] = paths['data_path']

        clf = params['clf'][1]

        attack(xdf, adv, attack_info, logger=logger)

if __name__ == '__main__':
    logger.log_print("END OF CODE")
    logger.log_close()
