import os
import sys
import re
import time
import pandas as pd
from tqdm import tqdm

path = r"C:\school\thesis\accuracy_and_ball_vs_clf_21022021"
dirs = [o for o in os.listdir(path)
        if os.path.isdir(os.path.join(path, o))]

metadata_sr = pd.Series()
metadata_sr['folders'] = 0
metadata_sr['Empty folders'] = 0
metadata_sr['Complete folders'] = 0

summary = pd.DataFrame(
    index=range(len(dirs)))
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
        summary.loc[idx, 'error_rate'] = r['1_minus_accuracy_before_attack']
        summary.loc[idx, 'accuracy_before'] = r['accuracy_before_attack']
        summary.loc[idx, 'accuracy_after'] = r['accuracy_after_attack']
        summary.loc[idx, 'clf'] = r['clf']
        summary.loc[idx, 'attack'] = r['attack']

        summary.loc[idx, 'prob_before'] = r['prob_of_adv_for_TRGT_before_attack']
        summary.loc[idx, 'prob_after'] = r['prob_of_adv_for_TRGT_after_attack']
        summary.loc[idx, 'success'] = r['prob_of_adv_for_TRGT_after_attack'] > 0.5

        summary.loc[idx, 'samples_in_ball_src'] = r['samples_in_ball_src']
        summary.loc[idx, 'samples_in_ball_trgt'] = r['samples_in_ball_trgt']
        summary.loc[idx, 'samples_in_ball_total'] = r['samples_in_ball_total']

        summary.loc[idx, 'samples_in_dataset_src'] = r['samples_in_dataset_src']
        summary.loc[idx, 'samples_in_dataset_trgt'] = r['samples_in_dataset_trgt']
        summary.loc[idx, 'samples_in_dataset_total'] = r['samples_in_dataset_total']

    else:
        metadata_sr['Empty folders'] += 1
        summary.loc[idx, 'run completed'] = False
        summary.loc[idx, 'uid'] = uid
        summary.loc[idx, 'attack'] = attack
        summary.loc[idx, 'clf'] = clf
time.sleep(0.1)

clfs = summary['clf'].unique()
attacks = summary['attack'].unique()
resdf = pd.DataFrame(index=range(len(clfs) * len(attacks)))

idx = -1
for attack in attacks:
    for clf in clfs:
        idx += 1
        xdf = summary[summary['attack'].eq(attack) & summary['clf'].eq(clf)]

        resdf.loc[idx, 'clf'] = clf
        resdf.loc[idx, 'clf error rate'] = xdf.error_rate.mean()
        resdf.loc[idx, 'src_in_ball_to_db'] = xdf['samples_in_ball_src'].mean() / xdf['samples_in_dataset_src'].mean()
        resdf.loc[idx, 'trgt_in_ball_to_db'] = xdf['samples_in_ball_trgt'].mean() / xdf[
            'samples_in_dataset_trgt'].mean()
        resdf.loc[idx, 'attack success rate'] = xdf['success'].astype(int).mean()
resdf.to_csv(os.path.join(r'C:\school\thesis\accuracy_and_ball_vs_clf_21022021', 'summary.csv'))
