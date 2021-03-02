import os


def build_batch_on_toy_only_clf(python_path, src_file, clf, attack='KNN', lambdas=[0.1], repeats=50):
    s_name = f'clf_{clf}_attack_{attack}_X_X.bat'

    s = ''
    s += '\n'
    s += '@echo off'
    for l in lambdas:
        s += '\n'
        s += f'{python_path} {src_file} --classifier {clf} --attack {attack} --lambda {l} -plt --run_id 0'
        s += '\n'
        s += f'for /L %%n in (1,1,{repeats}) do {python_path} {src_file} --classifier {clf} --attack {attack} --lambda {l} --run_id %%n'
    return s, s_name


if __name__ == '__main__':
    attacks = ['KNN']
    clfs = ['SVM', 'KNN5', 'ANN', 'DTree', 'Gaussian_NB']
    repeats = 50
    lambdas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    python_path = r'C:\Python37\python.exe'
    src_file = r'C:\school\thesis\omission\src\theory_experiments\accuracy_and_ball_vs_clf.py'
    output_path = r'C:\school\thesis\accuracy_and_ball_vs_clf'

    all_batches = list()
    for clf in clfs:
        for attack in attacks:
            s, s_name = build_batch_on_toy_only_clf(python_path=python_path,
                                                    src_file=src_file,
                                                    clf=clf,
                                                    attack=attack,
                                                    repeats=50,
                                                    lambdas=lambdas,
                                                    )
            output_batch = os.path.join(output_path, s_name)
            with open(output_batch, 'w+') as ffile:
                ffile.write(s)
                all_batches.append(output_batch)

    s = ''
    for p in all_batches[:-1]:
        s += f'start cmd /c "{p}"'
        s += '\n'
    s += f'start cmd /c "{all_batches[-1]}"'
    output_batch = os.path.join(output_path, 'run_all.bat')
    with open(output_batch, 'w+') as ffile:
        ffile.write(s)
        all_batches.append(output_batch)

    print(f"Output available in {output_path}")
