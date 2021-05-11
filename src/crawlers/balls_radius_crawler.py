import os
import sys
import re
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.crawlers.Crawler import FolderCrawler

if __name__ == '__main__':
    path = r"C:\school\thesis\accuracy_and_ball_vs_clf_19032021"
    crawler = FolderCrawler(path)
    crawler.scan()
    raw = crawler.get_raw()
    raw['epsilon_tag+'] = raw['epsilon_tag']
    raw['gamma_tag+epsilon'] = raw['gamma_tag'] + raw['epsilon']
    raw['gamma_tag+epsilon-epsilon_tag'] = raw['gamma_tag+epsilon'] - raw['epsilon_tag']

    csvpath = os.path.join(path, 'raw.csv')
    print(f"Results: {csvpath}")
    raw.to_csv(csvpath)

if __name__ == '__main__':
    summary = raw.copy()
    clfs = summary['clf'].unique()
    gammas = summary['gamma_tag'].value_counts().sort_index().index.to_list()

    # idx = -1
    # for clf in clfs:
    #     for gamma in gammas:
    #         if np.isnan(gamma):
    #             continue
    #         idx += 1
    #         clfdf = summary.loc[summary['clf'].eq(clf) & summary['gamma'].eq(gamma)]
    #
    #         sumdf.loc[idx, 'clf'] = clf
    #         sumdf.loc[idx, 'gamma'] = np.round(clfdf['gamma'].mean(), 6)
    #         sumdf.loc[idx, 'eps'] = clfdf['error_rate_epsilon'].mean()
    #         sumdf.loc[idx, 'success'] = clfdf['success'].mean()
    #         sumdf.loc[idx, 'acc_g_ball'] = clfdf['acc_g_ball_before'].mean()
    #         sumdf.loc[idx, 'acc_ghat_ball'] = clfdf['acc_g_ball_after'].mean()
    #         sumdf.loc[idx, 'count'] = clfdf['acc_g_ball_after'].count()
    #         sumdf.loc[idx, 'SRC points in ball'] = clfdf['src_in_ball'].mean().astype(int)
    #
    #         print(f"CLF: {clf}\tGamma: {gamma}\tCount: {clfdf.shape[0]}\tWins: {clfdf['success'].sum()}")
    #
    # csvpath = os.path.join(path, 'summary.csv')
    # print(f"Results: {csvpath}")
    # sumdf.to_csv(csvpath)
