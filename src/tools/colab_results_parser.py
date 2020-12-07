import re
import pandas as pd
import os

# src_dir = r'C:\school\thesis\omission\sarit_dsi_results_12-1-2020_victim-resnet18'
src_dir = r'C:\school\thesis\omission\sarit_dsi_results_12-5-2020_victim-alexnet'

if __name__ == '__main__':

    resdf = None
    section_parse_txt = True
    if section_parse_txt:
        src_txt = ''

        # from files
        files_txt = ''
        if src_dir is not None:
            paths = [p for p in os.listdir(src_dir) if '.txt' in p]
            for p in paths:
                path = os.path.join(src_dir, p)
                with open(path) as ffile:
                    files_txt += ffile.read()
        src_txt += files_txt

        pattern = r'([^@]*)@'
        m = re.findall(pattern, src_txt)
        m = [mt for mt in m if 'RES:' in mt]
        resdf = pd.DataFrame(index=range(len(m)),
                             columns=['execution_state', 'completed', 'win', 'accuracy_drop', 'random_seed', 'adv_idx',
                                      ])
        for idx, tm in enumerate(m):
            res = re.search('RES: ([a-zA-Z0-9]*)', tm).group(1)
            accdrop = float(re.search('Acc drop: ([-+]?[0-9]*\.?[0-9]+)', tm).group(1))
            pred_before = float(re.search('Prediction before: ([-+]?[0-9]*\.?[0-9]+)', tm).group(1))
            pred_after = float(re.search('Prediction after: ([-+]?[0-9]*\.?[0-9]+)', tm).group(1))
            random_seed = int(re.search('Random seed: ([-+]?[0-9]+)', tm).group(1))
            adv_idx = int(re.search('Adversarial sample: ([-+]?[0-9]+)', tm).group(1))

            resdf.loc[idx, 'execution_state'] = res
            resdf.loc[idx, 'completed'] = res != 'CANCELLED'
            resdf.loc[idx, 'win'] = res == 'WIN'
            resdf.loc[idx, 'accuracy_drop'] = accdrop
            resdf.loc[idx, 'random_seed'] = random_seed
            resdf.loc[idx, 'adv_idx'] = adv_idx

    section_analyse_df = True
    if section_analyse_df:
        num_of_sessions = resdf['completed'].count()
        completed_sessions = resdf['completed'].sum()
        session_completion_rate = float(completed_sessions) / num_of_sessions

        expdf = resdf[resdf['completed']]
        win_count = expdf['win'].sum()
        lose_count = completed_sessions - win_count
        win_rate = expdf['win'].sum() / expdf['win'].count()

        msg = ''
        msg += f'completed runs:\t {completed_sessions:>3} / {num_of_sessions:>3} ({100 * session_completion_rate:>.3f}%)' + '\n'
        msg += f'Wins          :\t {win_count:>3} / {completed_sessions:>3} ({100 * win_rate:>.3f} %)' + '\n'
        msg += f'Drop in accuracy: {100 * expdf["accuracy_drop"].mean():>.3f} %'
        print(msg)
