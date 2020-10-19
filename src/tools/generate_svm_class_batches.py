import os


def build_batch_with_src_and_trgt(python_path, src_file, clf, attack, trgt, src, repeats=50):
    s_name = f'clf_{clf}_attack_{attack}_{trgt}_{src}.bat'

    s = ''
    s += '\n'
    s += '@echo off'
    s += '\n'
    s += f'{python_path} {src_file} {clf} {attack} True 0 ({trgt},{src})'
    s += '\n'
    s += f'for /L %%n in (1,1,{repeats}) do {python_path} {src_file} {clf} {attack} False %%n ({trgt},{src})'
    return s, s_name


def build_batch_with_no_src_and_trgt(python_path, src_file, clf, attack, repeats=50):
    s_name = f'clf_{clf}_attack_{attack}_X_X.bat'

    s = ''
    s += '\n'
    s += '@echo off'
    s += '\n'
    s += f'{python_path} {src_file} {clf} {attack} True 0'
    s += '\n'
    s += f'for /L %%n in (1,1,{repeats}) do {python_path} {src_file} {clf} {attack} False %%n'
    return s, s_name


if __name__ == '__main__' and False:
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    attacks = ['Genetic', 'KNN']
    clfs = ['SVM']
    repeats = 50
    python_path = r'C:\Python37\python.exe'
    src_file = r'C:/school/thesis/omission/src/attacks_vs_clf_on_mnist_3labels.py'
    output_path = r'C:\school\thesis\clf vs learner MNIST 3LABEL'

    all_batches = list()
    for src in labels:
        for trgt in labels:
            if src == trgt:
                continue
            s, s_name = build_batch_with_src_and_trgt(python_path=python_path,
                                                      src_file=src_file,
                                                      clf=clfs[0],
                                                      attack=attacks[0],
                                                      trgt=trgt,
                                                      src=src,
                                                      repeats=50
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

if __name__ == '__main__':
    attacks = ['Genetic', 'KNN']
    clfs = ['SVM', 'KNN', 'ANN']
    repeats = 50
    python_path = r'C:\Python37\python.exe'
    src_file = r'C:/school/thesis/omission/src/attacks_vs_clf_on_mnist_3labels.py'
    output_path = r'C:\school\thesis\clf vs learner MNIST 3LABEL'

    all_batches = list()
    for clf in clfs:
        for attack in attacks:
            s, s_name = build_batch_with_no_src_and_trgt(python_path=python_path,
                                                         src_file=src_file,
                                                         clf=clf,
                                                         attack=attack,
                                                         repeats=50
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
