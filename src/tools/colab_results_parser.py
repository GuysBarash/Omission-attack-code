import re
import pandas as pd
import os

txt = '''
@
Random seed: 3203819323
Adversarial sample: 311
Prediction before: +0.014
Prediction after: -0.120
Acc drop: +0.005
RES: WIN
@
Random seed: 131146078
Adversarial sample: 5
Prediction before: +0.026
Prediction after: -0.109
Acc drop: +0.004
RES: WIN
@
Random seed: 4158010456
Adversarial sample: 346
Prediction before: -0.041
Prediction after: -0.176
Acc drop: -0.000
RES: CANCELLED EXP
@
Random seed: 3764185372
Adversarial sample: 160
Prediction before: +0.075
Prediction after: -0.053
Acc drop: -0.000
RES: WIN
@
Random seed: 3893912994
Adversarial sample: 316
Prediction before: +0.028
Prediction after: -0.100
Acc drop: +0.001
RES: WIN
@
Random seed: 2180069460
Adversarial sample: 185
Prediction before: +0.155
Prediction after: +0.012
Acc drop: -0.002
RES: LOSE
@
Random seed: 735203702
Adversarial sample: 638
Prediction before: -0.089
Prediction after: -0.212
Acc drop: -0.002
RES: CANCELLED EXP
@
Random seed: 780992882
Adversarial sample: 328
Prediction before: +0.196
Prediction after: +0.078
Acc drop: -0.003
RES: LOSE
@
Random seed: 66
Adversarial sample: 873
Prediction before: -0.039
Prediction after: -0.181
Acc drop: -0.006
RES: CANCELLED EXP
@
Random seed: 67
Adversarial sample: 699
Prediction before: +0.080
Prediction after: -0.047
Acc drop: -0.000
RES: WIN
@
Random seed: 68
Adversarial sample: 114
Prediction before: +0.156
Prediction after: +0.032
Acc drop: -0.001
RES: LOSE
@
Random seed: 69
Adversarial sample: 823
Prediction before: -0.078
Prediction after: +0.000
Acc drop: +0.941
RES: CANCELLED EXP
@
Random seed: 70
Adversarial sample: 324
Prediction before: +0.156
Prediction after: +0.036
Acc drop: -0.000
RES: LOSE
@
Random seed: 71
Adversarial sample: 891
Prediction before: -0.043
Prediction after: +0.000
Acc drop: +0.940
RES: CANCELLED EXP
@
Random seed: 72
Adversarial sample: 446
Prediction before: +0.027
Prediction after: -0.114
Acc drop: -0.003
RES: WIN
@
Random seed: 73
Adversarial sample: 702
Prediction before: +0.043
Prediction after: -0.080
Acc drop: +0.006
RES: WIN
@
Random seed: 74
Adversarial sample: 701
Prediction before: -0.006
Prediction after: +0.000
Acc drop: +0.940
RES: CANCELLED EXP
@
Random seed: 75
Adversarial sample: 719
Prediction before: -0.014
Prediction after: +0.000
Acc drop: +0.940
RES: CANCELLED EXP
@
Random seed: 76
Adversarial sample: 789
Prediction before: +0.008
Prediction after: -0.129
Acc drop: +0.006
RES: WIN
@
Random seed: 77
Adversarial sample: 884
Prediction before: +0.024
Prediction after: -0.113
Acc drop: +0.003
RES: WIN
@
Random seed: 78
Adversarial sample: 137
Prediction before: +0.126
Prediction after: -0.002
Acc drop: +0.003
RES: WIN
@
Random seed: 79
Adversarial sample: 129
Prediction before: +0.036
Prediction after: -0.095
Acc drop: +0.004
RES: WIN
@
Random seed: 80
Adversarial sample: 647
Prediction before: -0.080
Prediction after: +0.000
Acc drop: +0.940
RES: CANCELLED EXP
@
Random seed: 81
Adversarial sample: 188
Prediction before: +0.005
Prediction after: -0.120
Acc drop: -0.000
RES: WIN
@
'''
src_dir = r'C:\school\thesis\omission\sarit_dsi_results_12_1_2020'

if __name__ == '__main__':

    resdf = None
    section_parse_txt = True
    if section_parse_txt:
        src_txt = ''
        # from txt
        if txt is not None:
            src_txt += txt

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
