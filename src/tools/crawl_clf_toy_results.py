import os
import sys
import re
import time
import pandas as pd
from tqdm import tqdm
import itertools

# path = r"C:\school\thesis\clf vs learner"
path = r"C:\school\thesis\clf vs learner just clf"
dirs = [o for o in os.listdir(path)
        if os.path.isdir(os.path.join(path, o))]

metadata_sr = pd.Series()
metadata_sr['folders'] = 0
metadata_sr['Empty folders'] = 0
metadata_sr['Complete folders'] = 0

summary = pd.DataFrame(
    columns=['uid', 'run completed',
             'clf', 'accuracy',
             ],
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
        odf = pd.read_csv(csvpath).iloc[0]
        metadata_sr['Complete folders'] += 1

        summary.loc[idx, 'uid'] = uid
        summary.loc[idx, 'run completed'] = True
        summary.loc[idx, 'clf'] = odf['clf']
        summary.loc[idx, 'accuracy'] = odf['accuracy_before_attack']
        j = 3

    else:
        metadata_sr['Empty folders'] += 1
        summary.loc[idx, 'run completed'] = False
        summary.loc[idx, 'uid'] = uid
        summary.loc[idx, 'attack'] = attack
        summary.loc[idx, 'clf'] = clf
time.sleep(0.1)

# Metadata
msg = ''
msg += f"Successful experiments {metadata_sr['Complete folders']}/{metadata_sr['folders']}" + "\n"
print(msg)

summary['run completed'] = summary['run completed'].astype(bool)
failed_summary = summary.loc[~summary['run completed']].copy()
summary = summary.loc[summary['run completed']]

section_compare_clfs = True
if section_compare_clfs:
    clfs = summary['clf'].unique()
    df = pd.DataFrame(columns=['clf', 'accuracy mean', 'accuracy std',
                               'experiments count'
                               ], index=clfs)
    for clf in clfs:
        summary_t = summary.loc[summary['clf'].eq(clf)]
        df.loc[clf, 'clf'] = clf
        df.loc[clf, 'accuracy mean'] = summary_t['accuracy'].mean()
        df.loc[clf, 'accuracy std'] = summary_t['accuracy'].std()
        df.loc[clf, 'experiments count'] = summary_t.shape[0]

    df = df.sort_values(by=['accuracy mean'], ascending=False)
    print(df.head(10))
    summary_fname = 'clf_accuracy_compare.xlsx'
    summary_path = os.path.join(path, summary_fname)
    df.to_excel(summary_path)
    print(f"Accuracy file: {summary_fname}")
    print(f"Exported to: {path}")
