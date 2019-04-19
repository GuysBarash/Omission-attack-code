import os
import re
import pandas as pd
import numpy as np

start_path = r'C:\school\Thesa\30032019 RUNS'

if __name__ == '__main__':
    df = pd.DataFrame()

    fs = os.listdir(start_path)
    for fd in fs:
        pattern = r'(?i)([a-z0-9]+)_([0-9]+)'
        m = re.search(pattern, fd)
        if m is not None:
            ctactic = m.group(1)
            citer = int(m.group(2))
            fpath = os.path.join(start_path, fd)
            hardships_paths = os.listdir(fpath)
            hardships_paths = [p for p in hardships_paths if re.search('results_([0-9]+)', p) is not None]
            hardships_index = [int(re.search('results_([0-9]+)', p).group(1)) for p in hardships_paths]
            sr = pd.Series(name='{}_{}'.format(ctactic, citer),
                           index=hardships_index)
            df.loc[:, sr.name] = sr

            for hidx, hpath in enumerate(hardships_paths):
                full_hpath = os.path.join(fpath, hpath)
                relevant_files = os.listdir(full_hpath)
                current_hardship = hardships_index[hidx]
                win_budget = 999
                for possible_win_file in relevant_files:
                    m = re.search('A_VICTORY_AT_([0-9]+)', possible_win_file)
                    if m is not None:
                        win_budget = int(m.group(1))
                df.loc[current_hardship, sr.name] = win_budget
    df = df.sort_index()

    export_results = True
    if export_results:
        df = df.replace(999.0, np.nan)
        outpath = os.path.join(start_path, 'Summary.xlsx')
        tdf = df.transpose()
        tdf.to_excel(outpath)

        del outpath
        del export_results
