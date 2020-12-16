import re
import pandas as pd
import os
import tqdm

src_dir = r'C:\school\thesis\omission\results_after_benchmark\16-12-20_learner_alexnet_knn_googlenet'

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
                                      'budget', 'dataset size',
                                      'knn net', 'victim net',
                                      'run duration',
                                      'attacker net', 'victim net',
                                      'SRC class', 'TRGT class',
                                      ])
        for idx, tm in tqdm.tqdm(enumerate(m), total=len(m), desc='Reading jsons'):
            src_class = None
            res = re.search('RES: ([a-zA-Z0-9]*)', tm).group(1)
            accdrop = float(re.search('Acc drop: ([-+]?[0-9]*\.?[0-9]+)', tm).group(1))
            pred_before = (re.search('predicted class before: ([a-zA-Z]+)', tm).group(1))
            pred_after = (re.search('predicted class after: ([a-zA-Z]+)', tm).group(1))
            random_seed = int(re.search('Random seed: ([-+]?[0-9]+)', tm).group(1))
            adv_idx = int(re.search('Adversarial sample: ([-+]?[0-9]+)', tm).group(1))
            run_duration = float(re.search('duration: ([-+]?[0-9]*\.?[0-9]+)', tm).group(1))
            dataset_size = int(re.search('dataset size: ([-+]?[0-9]+)', tm).group(1))
            budget = int(re.search('Budget: ([-+]?[0-9]+)', tm).group(1))
            attack_net = re.search(r'knn net: ([a-zA-Z0-9]*)', tm).group(1)
            victim_net = re.search(r'Learner net: ([a-zA-Z0-9]*)', tm).group(1)
            src_class = re.findall('SRC class: ([a-zA-Z0-9]*)', tm)[0]
            trgt_class = re.findall('TRGT class: ([a-zA-Z0-9]*)', tm)[0]

            predictions_before = re.findall('prediction before ([a-zA-Z]+): ([-+]?[0-9]*\.?[0-9]+)', tm)
            predictions_before = {v1: float(v2) for v1, v2 in predictions_before}

            predictions_after = re.findall('prediction after ([a-zA-Z]+): ([-+]?[0-9]*\.?[0-9]+)', tm)
            predictions_after = {v1: float(v2) for v1, v2 in predictions_after}

            if idx == 0:
                for k in list(set(list(predictions_before.keys()) + list(predictions_after.keys()))):
                    col_name1 = f'{k}_before'
                    col_name2 = f'{k}_after'
                    if col_name1 not in resdf.columns:
                        resdf[col_name1] = 0
                    if col_name2 not in resdf.columns:
                        resdf[col_name2] = 0

            resdf.loc[idx, 'execution_state'] = res
            resdf.loc[idx, 'completed'] = res != 'CANCELLED'
            resdf.loc[idx, 'win'] = res == 'WIN'
            resdf.loc[idx, 'accuracy_drop'] = accdrop
            resdf.loc[idx, 'random_seed'] = random_seed
            resdf.loc[idx, 'adv_idx'] = adv_idx
            resdf.loc[idx, 'run duration'] = run_duration
            resdf.loc[idx, 'dataset size'] = dataset_size
            resdf.loc[idx, 'budget'] = budget
            resdf.loc[idx, 'attacker net'] = attack_net
            resdf.loc[idx, 'victim net'] = victim_net
            resdf.loc[idx, ['SRC class', 'TRGT class']] = src_class, trgt_class
            resdf.loc[idx, [f'{k}_before' for k in predictions_before.keys()]] = predictions_before.values()
            resdf.loc[idx, [f'{k}_after' for k in predictions_after.keys()]] = predictions_after.values()

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

        csvdir = os.path.dirname(__file__)
        csvpath = os.path.join(csvdir, 'Report.csv')
        resdf.to_csv(csvpath)
        print(f"Summary in: {csvdir}")
