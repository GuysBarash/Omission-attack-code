import os
import sys
import re
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

path = r"C:\school\thesis\accuracy_and_ball_vs_clf_03032021"
dirs = [o for o in os.listdir(path)
        if os.path.isdir(os.path.join(path, o))]

metadata_sr = pd.Series()
metadata_sr['folders'] = 0
metadata_sr['Empty folders'] = 0
metadata_sr['Complete folders'] = 0

summary = pd.DataFrame(
    index=range(len(dirs)))

if __name__ == '__main__' and False:

    time.sleep(0.1)
    for idx, odir in tqdm(enumerate(dirs), total=len(dirs)):
        metadata = odir.split('__')
        if len(metadata) == 3:
            attack, clf, uid = metadata
            has_plot = False
        elif len(metadata) == 4:
            attack, clf, uid, _ = metadata
            has_plot = True
        else:
            # Folder not relevant
            continue

        metadata_sr['folders'] += 1
        csvpath = os.path.join(os.path.join(path, odir), 'results.csv')
        if os.path.exists(csvpath):
            odf = pd.read_csv(csvpath)
            metadata_sr['Complete folders'] += 1
            r = odf.iloc[0]

            summary.loc[idx, 'uid'] = uid
            summary.loc[idx, 'run completed'] = True
            summary.loc[idx, 'clf'] = r['clf']
            summary.loc[idx, 'error_rate_epsilon'] = r['1_minus_accuracy_before_attack']
            summary.loc[idx, 'acc_g_ball_before'] = r['accuracy_before_attack_ball']
            summary.loc[idx, 'acc_g_ball_after'] = r['accuracy_before_attack_ball']
            summary.loc[idx, 'gamma'] = r['lambda']
            summary.loc[idx, 'ball_r'] = r['ball_r']
            summary.loc[idx, 'success'] = int(r['prob_of_adv_for_TRGT_after_attack'] > 0.5)
            summary.loc[idx, 'src_in_ball'] = r['samples_in_ball_src_train']

            # summary.loc[idx, 'accuracy_before'] = r['accuracy_before_attack']
            # summary.loc[idx, 'accuracy_after'] = r['accuracy_after_attack']
            # summary.loc[idx, 'clf'] = r['clf']
            # summary.loc[idx, 'attack'] = r['attack']
            #
            # summary.loc[idx, 'prob_before'] = r['prob_of_adv_for_TRGT_before_attack']
            # summary.loc[idx, 'prob_after'] = r['prob_of_adv_for_TRGT_after_attack']
            # summary.loc[idx, 'success'] = r['prob_of_adv_for_TRGT_after_attack'] > 0.5
            #
            # summary.loc[idx, 'samples_in_ball_src'] = r['samples_in_ball_src']
            # summary.loc[idx, 'samples_in_ball_trgt'] = r['samples_in_ball_trgt']
            # summary.loc[idx, 'samples_in_ball_total'] = r['samples_in_ball_total']
            #
            # summary.loc[idx, 'samples_in_dataset_src'] = r['samples_in_dataset_src']
            # summary.loc[idx, 'samples_in_dataset_trgt'] = r['samples_in_dataset_trgt']
            # summary.loc[idx, 'samples_in_dataset_total'] = r['samples_in_dataset_total']

        else:
            metadata_sr['Empty folders'] += 1
            summary.loc[idx, 'run completed'] = False
            summary.loc[idx, 'uid'] = uid
            summary.loc[idx, 'attack'] = attack
            summary.loc[idx, 'clf'] = clf
    time.sleep(0.1)

    csvpath = os.path.join(path, 'summary_raw.csv')
    print(f"Results: {csvpath}")
    summary.to_csv(csvpath)

csvpath = os.path.join(path, 'summary_raw.csv')
summary = pd.read_csv(csvpath, index_col=0)

clfs = list(summary['clf'].unique())
gammas = list(sorted(summary['gamma'].unique()))
sumdf = pd.DataFrame(columns=['clf', 'gamma', 'eps', 'success', 'acc_g_ball', 'acc_ghat_ball'],
                     index=range(len(clfs) * len(gammas)))

idx = -1
for clf in clfs:
    for gamma in gammas:
        if np.isnan(gamma):
            continue
        idx += 1
        clfdf = summary.loc[summary['clf'].eq(clf) & summary['gamma'].eq(gamma)]

        sumdf.loc[idx, 'clf'] = clf
        sumdf.loc[idx, 'gamma'] = np.round(clfdf['gamma'].mean(), 6)
        sumdf.loc[idx, 'eps'] = clfdf['error_rate_epsilon'].mean()
        sumdf.loc[idx, 'success'] = clfdf['success'].mean()
        sumdf.loc[idx, 'acc_g_ball'] = clfdf['acc_g_ball_before'].mean()
        sumdf.loc[idx, 'acc_ghat_ball'] = clfdf['acc_g_ball_after'].mean()
        sumdf.loc[idx, 'count'] = clfdf['acc_g_ball_after'].count()
        sumdf.loc[idx, 'SRC points in ball'] = clfdf['src_in_ball'].mean().astype(int)

        print(f"CLF: {clf}\tGamma: {gamma}\tCount: {clfdf.shape[0]}\tWins: {clfdf['success'].sum()}")

csvpath = os.path.join(path, 'summary.csv')
print(f"Results: {csvpath}")
sumdf.to_csv(csvpath)
