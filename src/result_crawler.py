import os
import sys
import re
import pandas as pd
from tqdm import tqdm

# path = r"C:\school\thesis\clf vs learner"
path = r"C:\school\thesis\clf vs learner MNIST"  # MNIST MODE
dirs = [o for o in os.listdir(path)
        if os.path.isdir(os.path.join(path, o))]

metadata_sr = pd.Series()
metadata_sr['folders'] = 0
metadata_sr['Empty folders'] = 0
metadata_sr['Complete folders'] = 0

summary = pd.DataFrame(
    columns=['uid', 'attack', 'clf', 'score before', 'score after', 'Success', 'acc before', 'acc after', 'acc drop'],
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
                            float(odf.iloc[0]['accuracy_before_attack']),
                            float(odf.iloc[0]['accuracy_after_attack']),
                            float(odf.iloc[0]['accuracy_before_attack']) - float(odf.iloc[0]['accuracy_after_attack']),
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
accuracy_drop_df = pd.DataFrame(columns=attacks, index=clfs)
for attack in attacks:
    t = summary[summary['attack'] == attack]
    if t.shape[0] == 0:
        continue
    for clf in t['clf'].unique():
        # if clf == 'ANN':
        #     continue
        curr_clf_df = t[t["clf"] == clf]
        score_after = curr_clf_df["score after"].mean()
        success_rate = curr_clf_df["Success"].mean()
        count = curr_clf_df["Success"].shape[0]

        positives_only_curr_clf_df = curr_clf_df[curr_clf_df['Success']]
        success_count = positives_only_curr_clf_df["Success"].shape[0]
        before_acc_mean = positives_only_curr_clf_df['acc before'].mean()
        before_in_acc_std = positives_only_curr_clf_df['acc before'].std()
        after_acc_mean = positives_only_curr_clf_df['acc after'].mean()
        after_in_acc_std = positives_only_curr_clf_df['acc after'].std()
        drop_in_acc_mean = positives_only_curr_clf_df['acc drop'].mean()
        drop_in_acc_std = positives_only_curr_clf_df['acc drop'].std()

        msg = ''
        msg += f'[ATTACK: {attack:>8}]'
        msg += f'[CLF: {clf:>15}]'
        msg += '\t'
        msg += f'[Instances: {count:> 4}]'
        msg += f'[Successful instances: {success_count:> 4}]'
        msg += '\t'
        msg += f'[Success {success_rate:>.4f}]'
        msg += '\t'
        # msg += f'[Prob. {score_after:>.4f}]'
        msg += f'[Accuracy: {before_acc_mean:>.4f}+{before_in_acc_std:>.4f} -->  {after_acc_mean:>.4f}+{after_in_acc_std:>.4f}]'

        msg += f'[Accuracy drop: {drop_in_acc_mean:>.4f}+{drop_in_acc_std:>.4f}]'
        edf.loc[clf, attack] = success_rate

        drop = u'{:>.3f}±{:>.3f}'.format(drop_in_acc_mean, drop_in_acc_std)
        accuracy_drop_df.loc[clf, attack] = drop
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

print("")
summary_fname = 'results.xlsx'
summary_path = os.path.join(path, summary_fname)
edf = edf[edf.columns.dropna()].dropna()
edf.to_excel(summary_path)
print(f"Summary file: {summary_fname}")
print(f"Exported to: {path}")

summary_fname = 'accuracy drop.xlsx'
summary_path = os.path.join(path, summary_fname)
accuracy_drop_df = accuracy_drop_df[accuracy_drop_df.columns.dropna()].dropna()
accuracy_drop_df.to_excel(summary_path)
print(f"Accuracy file: {summary_fname}")
print(f"Exported to: {path}")
