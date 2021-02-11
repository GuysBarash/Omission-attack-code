import os
import sys
import re
import time
import pandas as pd
from tqdm import tqdm
import itertools

# path = r"C:\school\thesis\clf vs learner"
path = r"C:\school\thesis\clf vs learner wb vs bb mnist 05-01-2021"  # MNIST MODE
dirs = [o for o in os.listdir(path)
        if os.path.isdir(os.path.join(path, o))]

metadata_sr = pd.Series()
metadata_sr['folders'] = 0
metadata_sr['Empty folders'] = 0
metadata_sr['Complete folders'] = 0

summary = pd.DataFrame(
    columns=['uid', 'run completed', 'src label', 'trgt label',
             'attack',
             'clf', 'score before',
             'score after (WB)', 'score after (BB)',
             'Success (WB)', 'Success (BB)',
             'acc before',
             'acc after (WB)', 'acc after (BB)',
             'acc drop'],
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

        summary.loc[idx, 'uid'] = uid
        summary.loc[idx, 'run completed'] = True
        summary.loc[idx, 'src label'] = odf.iloc[0]['SRC number']
        summary.loc[idx, 'trgt label'] = odf.iloc[0]['TRGT number']
        summary.loc[idx, 'attack'] = attack
        summary.loc[idx, 'clf'] = clf
        summary.loc[idx, 'score before'] = float(odf.iloc[0]['prob_of_adv_for_TRGT_before_attack'])
        summary.loc[idx, 'score after (WB)'] = float(odf.iloc[0]['prob_of_adv_for_TRGT_after_attack'])
        summary.loc[idx, 'score after (BB)'] = float(odf.iloc[0]['prob_of_adv_for_TRGT_after_attack_BB'])
        summary.loc[idx, 'Success (WB)'] = float(odf.iloc[0]['prob_of_adv_for_TRGT_after_attack']) >= 0.5
        summary.loc[idx, 'Success (BB)'] = float(odf.iloc[0]['prob_of_adv_for_TRGT_after_attack_BB']) >= 0.5
        summary.loc[idx, 'acc before'] = float(odf.iloc[0]['accuracy_before_attack'])
        summary.loc[idx, 'acc after (WB)'] = float(odf.iloc[0]['accuracy_after_attack'])
        summary.loc[idx, 'acc after (BB)'] = float(odf.iloc[0]['accuracy_after_attack'])
        summary.loc[idx, 'acc drop'] = float(odf.iloc[0]['accuracy_before_attack']) - float(
            odf.iloc[0]['accuracy_after_attack'])
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
    attacks = summary['attack'].unique()
    clfs = summary['clf'].unique()
    sigs = {f'{attack}_{clf}': (attack, clf) for attack, clf in itertools.product(attacks, clfs)}
    df = pd.DataFrame(columns=['attack', 'clf',
                               'score after (WB) MEAN', 'score after (WB) STD',
                               'score after (BB) MEAN', 'score after (BB) STD',
                               'Success rate (WB) MEAN', 'Success rate (WB) STD',
                               'Success rate (BB) MEAN', 'Success rate (BB) STD',
                               'experiments count'
                               ], index=sigs.keys())
    for sig, (attack, clf) in sigs.items():
        summary_t = summary.loc[summary['attack'].eq(attack) & summary['clf'].eq(clf)]
        df.loc[sig, 'attack'] = attack
        df.loc[sig, 'clf'] = clf
        df.loc[sig, 'score after (WB) MEAN'] = summary_t['score after (WB)'].mean()
        df.loc[sig, 'score after (WB) STD'] = summary_t['score after (WB)'].std()
        df.loc[sig, 'Success rate (WB) MEAN'] = summary_t['Success (WB)'].mean()
        df.loc[sig, 'Success rate (WB) STD'] = summary_t['Success (WB)'].std()
        df.loc[sig, 'score after (BB) MEAN'] = summary_t['score after (BB)'].mean()
        df.loc[sig, 'score after (BB) STD'] = summary_t['score after (BB)'].std()
        df.loc[sig, 'Success rate (BB) MEAN'] = summary_t['Success (BB)'].mean()
        df.loc[sig, 'Success rate (BB) STD'] = summary_t['Success (BB)'].std()
        df.loc[sig, 'experiments count'] = summary_t.shape[0]

    summary_fname = 'attackVSclf.xlsx'
    summary_path = os.path.join(path, summary_fname)
    df.to_excel(summary_path)
    print(f"Accuracy file: {summary_fname}")
    print(f"Exported to: {path}")

section_compare_all_labels = False
if section_compare_all_labels:
    summary['src_trgt'] = summary['src label'].astype(str) + '_' + summary['trgt label'].astype(str)
    g = summary.groupby(by=['src_trgt'])

    from_labels = summary['src label'].unique().astype(int)
    from_labels.sort()
    to_labels = summary['trgt label'].unique().astype(int)
    to_labels.sort()
    df_template = pd.DataFrame(index=from_labels, columns=to_labels)

    result_acc_drop_mean_df = df_template.copy()
    result_acc_drop_std_df = df_template.copy()
    result_success_mean_df = df_template.copy()
    result_success_std_df = df_template.copy()

    unique_keys = summary['src_trgt'].unique()
    time.sleep(0.1)
    for k in tqdm(unique_keys, total=unique_keys.shape[0], desc='Comparing labels'):
        df = summary[summary['src_trgt'].eq(k)]
        from_label = df.iloc[0]['src label']
        to_label = df.iloc[0]['trgt label']
        result_acc_drop_mean_df.loc[from_label, to_label] = df['acc drop'].mean()
        result_acc_drop_std_df.loc[from_label, to_label] = df['acc drop'].std()
        result_success_mean_df.loc[from_label, to_label] = df['Success (WB)'].mean()
        result_success_std_df.loc[from_label, to_label] = df['Success (WB)'].std()
    time.sleep(0.1)

    print("")
    summary_fname = 'success rate MEAN.xlsx'
    summary_path = os.path.join(path, summary_fname)
    result_success_mean_df.to_excel(summary_path)
    print(f"Summary file: {summary_fname}")
    print(f"Exported to: {path}")

    print("")
    summary_fname = 'success rate STD.xlsx'
    summary_path = os.path.join(path, summary_fname)
    result_success_std_df.to_excel(summary_path)
    print(f"Summary file: {summary_fname}")
    print(f"Exported to: {path}")

    summary_fname = 'accuracy drop MEAN.xlsx'
    summary_path = os.path.join(path, summary_fname)
    result_acc_drop_mean_df.to_excel(summary_path)
    print(f"Accuracy file: {summary_fname}")
    print(f"Exported to: {path}")

    summary_fname = 'accuracy drop STD.xlsx'
    summary_path = os.path.join(path, summary_fname)
    result_acc_drop_std_df.to_excel(summary_path)
    print(f"Accuracy file: {summary_fname}")
    print(f"Exported to: {path}")
