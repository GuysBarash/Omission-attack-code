import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from tools.paths import itemized as paths

if __name__ == '__main__':
    workpath = paths['work_path']
    df = pd.DataFrame(columns=['FROM', 'TO', 'ITER', 'path', 'full_path', 'START SCORE', 'END SCORE'])
    for fld in os.listdir(workpath):
        pattern = r'from_([0-9]+)_to_([0-9]+)_iter_([0-9]+)'
        m = re.search(pattern, fld)
        if m is not None:
            sr = pd.Series()
            sr['FROM'] = int(m.group(1))
            sr['TO'] = int(m.group(2))
            sr['ITER'] = int(m.group(3))
            sr['path'] = fld
            sr['full_path'] = os.path.join(workpath, fld)
            df = df.append(sr, ignore_index=True)

    # Fetching score from files
    for row_idx, row in tqdm(df.iterrows(), total=df.shape[0], desc='Fetching \'per_round_winner\' CSV files'):
        csv_path = os.path.join(row['full_path'], 'per_round_winner.csv')
        xdf = pd.read_csv(csv_path)
        score = xdf.prob_of_target.iloc[-1]
        df.loc[row_idx, 'END SCORE'] = score

    # Build success matrix
    from_classes = sorted(df['FROM'].unique())
    to_classes = sorted(df['TO'].unique())
    results_df = pd.DataFrame(columns=to_classes, index=from_classes, dtype=np.float)
    for from_class in from_classes:
        for to_class in to_classes:
            xdf = df[(df['FROM'] == from_class) & (df['TO'] == to_class)]
            if len(xdf) == 0:
                score = np.NaN
            else:
                score = xdf['END SCORE'].mean()
            results_df.loc[from_class, to_class] = score

    rdf = results_df.copy()
    rdf['AVG'] = results_df.mean(axis=1)
    rdf.loc['AVG'] = results_df.mean(axis=0)
    rdf.loc['AVG', 'AVG'] = results_df.mean(axis=0).mean()
    rdf = rdf.fillna(1.0)
    rdf = rdf.reindex(index=rdf.index[::-1])

    sns.heatmap(rdf, annot=True)
    plt.hlines(1, -1, 11, linewidth=5, color='white')
    plt.vlines(10, -1, 11, linewidth=5, colors='white')
    png_path = os.path.join(paths['data_path'], 'Heatmap.png')
    plt.savefig(png_path)
    csv_path = os.path.join(paths['data_path'], 'Heatmap.csv')
    rdf.to_csv(csv_path)
