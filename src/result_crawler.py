import os
import sys
import re
import pandas as pd
from tqdm import tqdm

path = r'C:\school\thesis\clf vs learner MNIST_12072020'
dirs = [o for o in os.listdir(path)
        if os.path.isdir(os.path.join(path, o))]

metadata_sr = pd.Series()
metadata_sr['folders'] = 0
metadata_sr['Empty folders'] = 0
metadata_sr['Complete folders'] = 0

summary = pd.DataFrame(columns=['uid', 'attack', 'clf', 'score before', 'score after', 'Success'],
                       index=range(len(dirs)))
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
        summary.loc[idx] = [uid, attack, clf,
                            float(odf.iloc[0]['prob_of_adv_for_TRGT_before_attack']),
                            float(odf.iloc[0]['prob_of_adv_for_TRGT_after_attack']),
                            float(odf.iloc[0]['prob_of_adv_for_TRGT_after_attack']) >= 0.5,
                            ]
    else:
        metadata_sr['Empty folders'] += 1

# Metadata
msg = ''
msg += f"Successful experiments {metadata_sr['Complete folders']}/{metadata_sr['folders']}" + "\n"
print(msg)

attacks = summary['attack'].unique()
clfs = summary['clf'].unique()
edf = pd.DataFrame(columns=attacks, index=clfs)
for attack in attacks:
    t = summary[summary['attack'] == attack]
    if t.shape[0] == 0:
        continue
    for clf in t['clf'].unique():
        # if clf == 'ANN':
        #     continue

        score_after = t[t["clf"] == clf]["score after"].mean()
        success_rate = t[t["clf"] == clf]["Success"].mean()
        count = t[t["clf"] == clf]["Success"].shape[0]
        msg = ''
        msg += f'[ATTACK: {attack:>8}]'
        msg += f'[CLF: {clf:>15}]'
        msg += f'[Instances: {count:> 4}]'
        msg += '\t'
        msg += f'[Success {success_rate:>.4f}]'
        # msg += f'[Prob. {score_after:>.4f}]'
        edf.loc[clf, attack] = success_rate
        print(msg)
    print("")

    count = t["Success"].shape[0]
    success_rate = t["Success"].mean()
    msg = ''
    msg += f'[ATTACK: {attack:>8}]'
    msg += f'[Instances: {count:> 4}]'
    msg += '\t'
    msg += f'[Success {success_rate:>.4f}]'
    print(msg)
    print('')

summary_fname = 'results.xlsx'
summary_path = os.path.join(path, summary_fname)
edf = edf[edf.columns.dropna()].dropna()
edf.to_excel(summary_path)
print("")
print(f"Summary file: {summary_fname}")
print(f"Exported to: {path}")
