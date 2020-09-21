import os
import sys
import re
import time
import pandas as pd
from tqdm import tqdm

# path = r"C:\school\thesis\clf vs learner"
path = r"C:\school\thesis\clf labels compare MNIST"  # MNIST MODE
dirs = [o for o in os.listdir(path)
        if os.path.isdir(os.path.join(path, o))]

metadata_sr = pd.Series()
metadata_sr['folders'] = 0
metadata_sr['Empty folders'] = 0
metadata_sr['Complete folders'] = 0

summary = pd.DataFrame(
    columns=['uid', 'src label', 'trgt label', 'attack', 'clf', 'score before', 'score after', 'Success', 'acc before',
             'acc after', 'acc drop'],
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
        summary.loc[idx] = [uid,
                            odf.iloc[0]['SRC number'], odf.iloc[0]['TRGT number'],
                            attack, clf,
                            float(odf.iloc[0]['prob_of_adv_for_TRGT_before_attack']),
                            float(odf.iloc[0]['prob_of_adv_for_TRGT_after_attack']),
                            float(odf.iloc[0]['prob_of_adv_for_TRGT_after_attack']) >= 0.5,
                            float(odf.iloc[0]['accuracy_before_attack']),
                            float(odf.iloc[0]['accuracy_after_attack']),
                            float(odf.iloc[0]['accuracy_before_attack']) - float(odf.iloc[0]['accuracy_after_attack']),
                            ]
    else:
        metadata_sr['Empty folders'] += 1
time.sleep(0.1)

# Metadata
msg = ''
msg += f"Successful experiments {metadata_sr['Complete folders']}/{metadata_sr['folders']}" + "\n"
print(msg)

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
    result_success_mean_df.loc[from_label, to_label] = df['Success'].mean()
    result_success_std_df.loc[from_label, to_label] = df['Success'].std()
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
