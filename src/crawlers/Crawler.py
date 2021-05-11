import pandas as pd
import numpy as np

from tqdm import tqdm

import os
import sys
import time


class FolderCrawler:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.rawdata = None
        self.metadata = None
        self.rawdata_export_path = None
        self.metadata_export_path = None

    def scan(self, data_dir=None, export=True, skip_if_exist=True):
        if data_dir is not None:
            self.data_dir = data_dir
            self.rawdata = None
            self.metadata = None
        if self.data_dir is None:
            raise Exception("No path was given. Please provide path either in init or in scan")
        else:
            self.rawdata_export_path = os.path.join(self.data_dir, 'rawdata.csv')
            self.metadata_export_path = os.path.join(self.data_dir, 'metadata.csv')

        if skip_if_exist:
            if os.path.exists(self.rawdata_export_path) and os.path.exists(self.metadata_export_path):
                print("Scan already completed. Skipping.")
                self.metadata = pd.read_csv(self.metadata_export_path, squeeze=True, index_col=0)
                self.rawdata = pd.read_csv(self.rawdata_export_path, index_col=0)
                print("Run completed.")
                print(f"Folders scanned: {self.metadata['Complete folders']}/{self.metadata['folders']}")
                print(f"Data in: {self.data_dir}")
                return None

        dirs = [o for o in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, o))]

        self.metadata = pd.Series()
        self.metadata['folders'] = 0
        self.metadata['Empty folders'] = 0
        self.metadata['Complete folders'] = 0

        self.rawdata = None
        time.sleep(0.1)
        for idx, odir in tqdm(enumerate(dirs), total=len(dirs)):
            self.metadata['folders'] += 1

            csvpath = os.path.join(os.path.join(path, odir), 'results.csv')
            if os.path.exists(csvpath):
                odf = pd.read_csv(csvpath)
                self.metadata['Complete folders'] += 1
                if self.rawdata is None:
                    self.rawdata = pd.DataFrame(index=range(len(dirs)),
                                                columns=odf.columns
                                                )

                r = odf.iloc[0]
                self.rawdata.loc[idx] = r

            else:
                self.metadata['Empty folders'] += 1
        time.sleep(0.1)

        if export:
            self.metadata.to_csv(self.metadata_export_path, header=True)
            self.rawdata.to_csv(self.rawdata_export_path, index=0)
            print("Run completed.")
            print(f"Folders scanned: {self.metadata['Complete folders']}/{self.metadata['folders']}")
            print(f"Data in: {self.data_dir}")

        return None

    def get_raw(self):
        return self.rawdata

    def get_meta(self):
        return self.metadata


if __name__ == '__main__':
    path = r"C:\school\thesis\accuracy_and_ball_vs_clf_19032021"
    crawler = FolderCrawler(path)
    crawler.scan()
    raw = crawler.get_raw()
