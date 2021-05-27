import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil


def verify_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return True


def clear_path(path):
    if os.path.exists(path):
        all_items_to_remove = [os.path.join(path, f) for f in os.listdir(path)]
        for item_to_remove in all_items_to_remove:
            if os.path.exists(item_to_remove) and not os.path.isdir(item_to_remove):
                os.remove(item_to_remove)
            else:
                shutil.rmtree(item_to_remove)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_test_idx(labels):
    arr = np.array(labels)
    ret = dict()
    for l in np.unique(arr):
        idxs = np.where(arr == l)[0]
        test_idxs = np.random.choice(idxs, int(np.floor(idxs.shape[0] * 0.1)), replace=False)
        ret[l] = test_idxs
    return ret


src_path = r'C:\Users\32519\Downloads\cifar-10-python.tar\cifar-10-python\cifar-10-batches-py'
f = [os.path.join(src_path, ft) for ft in os.listdir(src_path) if 'data_batch' in ft]
trgt_path = r'C:\school\thesis\omission\CIFAR_FULL'
train_path = r'C:\school\thesis\omission\CIFAR_FULL\train'
test_path = r'C:\school\thesis\omission\CIFAR_FULL\test'
counter = dict()
clear_path(trgt_path)

train_count = dict()
test_count = dict()
for fidx, ft in enumerate(f):
    d = unpickle(ft)
    imgs_count = len(d[b'labels'])
    test_idxs = get_test_idx(d[b'labels'])
    for idx in tqdm(range(imgs_count), desc=f'Unzipping batch {fidx}'):
        label = d[b'labels'][idx]
        img = d[b'data'][idx]
        img = np.swapaxes(img.reshape(3, 32, 32), 0, 2)
        img = Image.fromarray(img, 'RGB')
        img = img.rotate(-90)

        img_idx = counter.get(label, 0)
        counter[label] = img_idx + 1
        if idx in test_idxs[label]:
            path = os.path.join(test_path, str(label))
            test_count[label] = test_count.get(label, 0) + 1
        else:
            path = os.path.join(train_path, str(label))
            train_count[label] = train_count.get(label, 0) + 1
        verify_path(path)
        img_name = os.path.join(path, f'{img_idx:04}.jpg')
        img.save(img_name, "JPEG")

for k in train_count.keys():
    print(f"[Label: {k}]Train: {train_count[k]}\tTest: {test_count[k]}")
