import os
import sys
import shutil
import time
import re
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

src_path = r'C:\school\thesis\omission\TinyImageNet\tiny-imagenet-200'
trgt_path = r'C:\school\thesis\omission\TinyImageNet\imgnet_data'


def clear_folder(path, clear_if_exist=False):
    if os.path.exists(path) and clear_if_exist:
        all_items_to_remove = [os.path.join(path, f) for f in os.listdir(path)]
        for item_to_remove in all_items_to_remove:
            if os.path.exists(item_to_remove) and not os.path.isdir(item_to_remove):
                os.remove(item_to_remove)
            else:
                shutil.rmtree(item_to_remove)

    if not os.path.exists(path):
        os.makedirs(path)


def _copy_file(k):
    src = k['src']
    trgt = k['trgt']
    if os.path.isfile(src):
        shutil.copyfile(src, trgt)
    else:
        shutil.copytree(src, trgt)


def copy_files(l, concurrent=False):
    inps = list()
    for idx, lt in enumerate(l):
        d = dict()
        d['idx'] = idx
        d['src'] = lt[0]
        d['trgt'] = lt[1]
        inps.append(d)

    time.sleep(0.1)
    if concurrent:
        bar = tqdm(inps, desc='Concurrent copy', total=len(inps))
        for dt in bar:
            _copy_file(dt)
    else:
        bar = tqdm(inps, desc='Parallel copy', total=len(inps))
        with Pool(processes=4) as pool:
            p = pool.imap(_copy_file, inps)
            for dt in bar:
                p.next()
    time.sleep(0.1)


if __name__ == '__main__':
    section_locate_legend = False
    if section_locate_legend:
        lpath = os.path.join(src_path, 'words.txt')
        with open(lpath) as ffile:
            txt = ffile.read()
            df = pd.DataFrame(data=[t.split('\t') for t in txt.split('\n')]).set_index(keys=[0])
            df[1] = df[1].str.replace(r'\W+', '_')
            legendf = df[1]

    section_copy = False
    if section_copy:
        clear_folder(trgt_path, False)
        section_copy_train = False
        if section_copy_train:
            ftype = 'train'
            print(f"Copying {ftype} files")
            src_sub_path = os.path.join(src_path, ftype)
            trgt_sub_path = os.path.join(trgt_path, ftype)
            clear_folder(trgt_sub_path, True)
            dirs = os.listdir(src_sub_path)
            labels = [legendf[t] for t in dirs]
            dirs_map = [(
                os.path.join(src_sub_path, dirs[i], 'images'),
                os.path.join(trgt_sub_path, labels[i])
            )
                for i in
                range(len(dirs))]
            copy_files(dirs_map)

        section_copy_test = False
        if section_copy_test:
            ftype = 'test'
            print(f"Copying {ftype} files")
            src_sub_path = os.path.join(src_path, 'val', 'images')
            trgt_sub_path = os.path.join(trgt_path, ftype)
            clear_folder(trgt_sub_path, True)
            # Translate names to labels
            lpath = os.path.join(src_path, 'val', 'val_annotations.txt')
            with open(lpath) as ffile:
                mapdf = pd.read_csv(lpath, sep='\t', header=None)
            mapdf = mapdf[[0, 1]]
            dirs_map = list()
            for label_code in mapdf[1].unique():
                label = legendf[label_code]
                trgt_subsub_path = os.path.join(trgt_sub_path, label)
                clear_folder(trgt_subsub_path, False)
                mapdf_t = mapdf[mapdf[1].eq(label_code)]
                dirs_map += [(
                    os.path.join(src_sub_path, mapdf_t.iloc[i, 0]),
                    os.path.join(trgt_subsub_path, mapdf_t.iloc[i, 0])
                )
                    for i in
                    range(mapdf_t.shape[0])]
            copy_files(dirs_map)

    section_compare_data = True
    if section_compare_data:
        train_path = os.path.join(trgt_path, 'train')
        test_path = os.path.join(trgt_path, 'test')

        idxs = list(set(os.listdir(train_path) + os.listdir(test_path)))
        comparedf = pd.DataFrame(index=idxs, columns=['train_count', 'test_count'], data=0)

        for k in tqdm(idxs, desc='Counting files'):
            train_path_k = os.path.join(train_path, k)
            count_train = 0
            if os.path.exists(train_path_k):
                count_train = len(os.listdir(train_path_k))
            test_path_k = os.path.join(test_path, k)
            count_test = 0
            if os.path.exists(test_path_k):
                count_test = len(os.listdir(test_path_k))
            comparedf.loc[k] = [count_train, count_test]
