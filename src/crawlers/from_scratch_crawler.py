import pandas as pd
import numpy as np
import re

from tqdm import tqdm

import os
import sys
import time


class FolderCrawler:
    def extract(self, fname):
        with open(fname) as ffile:
            l = ffile.read()
            ls = l.split('\n')[:-1]
            sr = pd.Series()
            for ls_item in ls:
                k, v = ls_item.split(':')
                v = re.sub(r'\s', '', v)
                if k == 'Attack duration' and k in sr:
                    k = 'Training duration'
                if 'duration' in k:
                    v = float(v)

                sr[k] = v
        return sr

    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.rawdata = None
        self.metadata = None
        self.rawdata_export_path = None
        self.metadata_export_path = None

    def scan(self, data_dir=None, export=True, use_if_exists=True):
        if data_dir is not None:
            self.data_dir = data_dir
            self.rawdata = None
        if self.data_dir is None:
            raise Exception("No path was given. Please provide path either in init or in scan")
        else:
            self.rawdata_export_path = os.path.join(self.data_dir, 'rawdata.csv')

        if use_if_exists and os.path.exists(self.rawdata_export_path):
            print("Raw data already exist, loading from memory")
            self.rawdata = pd.read_csv(self.rawdata_export_path)
        else:
            logfiles = [fs for fs in os.listdir(self.data_dir) if len(re.findall(r'Report_[0-9].*\.txt', fs)) > 0]
            for fx in tqdm(logfiles, desc='Reading logs'):
                full_path = os.path.join(self.data_dir, fx)
                sr = self.extract(full_path)
                sr['file'] = fx
                sr['path'] = full_path
                sr['win'] = int(sr['RES'] == 'WIN')
                sr['lose'] = int(sr['RES'] == 'LOSE')

                if self.rawdata is None:
                    self.rawdata = pd.DataFrame(columns=sr.index, index=logfiles)
                self.rawdata.loc[fx] = sr

            if export:
                self.rawdata.to_csv(self.rawdata_export_path, index=0)
                print("Run completed.")
                print(f"Raw Data in: {self.data_dir}")

        return None

    def extract_metadata(self, export=True):
        baseline_idxs = self.rawdata['Budget'].astype(int) == 0
        self.baseline = self.rawdata[baseline_idxs]
        self.rawdata = self.rawdata[~baseline_idxs]
        baseline_accuracy = self.baseline['Acc after'].astype(float).mean()

        self.metadata = pd.Series(index=['Instances', 'Wins', 'win rate',
                                         'Accuracy', 'Accuracy std',
                                         'Budget', 'Dataset size',
                                         'Total duration',
                                         'Attack duration', 'Training duration', 'train-to-attack ratio'])

        self.metadata['Instances'] = self.rawdata['Random seed'].count()
        self.metadata['Wins'] = self.rawdata['win'].sum()
        self.metadata['win rate'] = self.metadata['Wins'] / self.metadata['Instances']
        self.metadata['Budget'] = self.rawdata['Budget'].astype(int).mean()
        self.metadata['Accuracy'] = self.rawdata['Acc after'].astype(float).mean()
        self.metadata['Accuracy std'] = self.rawdata['Acc after'].astype(float).std()
        self.metadata['Accuracy diff from baseline'] = baseline_accuracy - self.metadata['Accuracy']
        self.metadata['Dataset size'] = int(self.rawdata['dataset size'].astype(int).mean())
        self.metadata['Total duration'] = np.round(self.rawdata['Total duration'].mean() / 3600, 2)
        self.metadata['Training duration'] = np.round(self.rawdata['Training duration'].mean() / 3600, 2)
        self.metadata['Attack duration'] = np.round(self.rawdata['Attack duration'].mean() / 3600, 2)
        self.metadata['train-to-attack ratio'] = np.round(
            self.metadata['Attack duration'] / self.metadata['Training duration'], 2)

        for k, v in self.metadata.items():
            print(f"[{k}]\t{v:>.3f}")

        if export:
            self.metadata.to_csv(self.metadata_export_path, header=True)
            print("Run completed.")
            print(f"MetaData in: {self.data_dir}")

    def get_raw(self):
        return self.rawdata

    def get_meta(self):
        return self.metadata


if __name__ == '__main__':
    path = r"C:\school\thesis\vision_transfer_budgets_13032021"
    crawler = FolderCrawler(path)
    crawler.scan()
    raw = crawler.get_raw()
    j = 3
