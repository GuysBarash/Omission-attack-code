# -*- coding: utf-8 -*-
"""pytorch_knn_attack.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rRlyAFROAQ09X3unEKRcHo5RYx2ap_5o
"""

import gc
import os
import shutil
import sys
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from pynvml import *
from IPython.display import display
from PIL import Image
from tabulate import tabulate

import argparse

DEBUG_MODE = False
learner_type = 'resnet18'
knn_detector_type = 'googlenet'
thread_name = 'First'
train_epocs = 70
train_batch_size = 128
random_seed_start_index = 1150
experiments_to_run = 300


def pretty_print_dict(d):
    msg = ''
    win_key = max(d, key=d.get)
    for k, v in d.items():
        tg = ''
        if k == win_key:
            tg = ' <--'
        msg += f'[{k:<10}]\t{v:>.3f}{tg}' + '\n'
    return msg


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


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    t0 = preds.int() == labels.int()
    t1 = torch.sum(t0)
    t1 = t1.item()
    t2 = len(preds)
    t3 = t1 / t2
    t3 = torch.tensor(t3)
    return t3


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()


def debug_memory(title=None, clear_mem=True):
    # print("<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if title is not None:
        # print(title)
        pass

    if torch.cuda.is_available():
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        free_mem = info.used
        MB = 2 ** 20
        msg = f'GPU Mem: {info.used/MB:,.1f}/{info.total/MB:,.1f} [{100 * float(info.used)/info.total:>.3f}%]'
    else:
        msg = 'No GPU. Running on CPU.'
    # print(msg)

    # print("<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


class KNNDetector:
    def __init__(self, transform, model_type='alexnet'):
        # Load the pretrained model
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.transform = transform
        self.layer = None
        self.model_type = model_type

    def reset(self):
        if self.model_type == 'alexnet':
            self.model = models.alexnet(pretrained=True)
            self.layer = self.model._modules.get('avgpool')
            self.size_of_vector = 9216
        elif self.model_type == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.layer = self.model._modules.get('avgpool')
            self.size_of_vector = self.model.fc.in_features
        elif self.model_type == 'googlenet':
            self.model = models.googlenet(pretrained=True)
            self.layer = self.model._modules.get('avgpool')
            self.size_of_vector = self.model.fc.in_features

        else:
            exit(1)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def get_vector(self, image_path):
        # 1. Load the image with Pillow library
        img = Image.open(image_path)

        # 2. Create a PyTorch Variable with the transformed image
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        batch_t = batch_t.to(self.device)

        my_embedding = torch.zeros([batch_t.shape[0], self.size_of_vector, batch_t.shape[0], batch_t.shape[0]])

        def copy_data(m, i, o):
            # my_embedding.copy_(o.data)  # ResNet
            my_embedding.copy_(o.data.reshape(*my_embedding.shape))

        # 5. Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)

        # Predict
        self.model.eval()
        out = self.model(batch_t)

        # outvec = torch.nn.functional.softmax(out[0], dim=0)
        # leading_idx = list(zip(outvec.sort()[1][-10:], outvec.sort()[0][-10:]))
        # leading_labels = [idx2label(int(idx[0])) for idx in leading_idx]
        # leading_p = [100 * float(idx[1]) for idx in leading_idx]
        # pred = pd.Series(index=leading_labels, data=leading_p)

        h.remove()
        embd = my_embedding[0, :, 0, 0]

        del batch_t
        return embd.clone()

    def get_vectors(self, paths, batch_size=20):
        # 1. Load the image with Pillow library
        imgs_t = list()
        for img_path in tqdm(paths, total=len(paths), desc='KNNDetecor opening imgs'):
            with Image.open(img_path) as o_img:
                imgs_t.append(self.transform(o_img))

        # 2. Create a PyTorch Variable with the transformed image
        ret = None
        iterator = np.arange(0, len(imgs_t), batch_size)
        for batch_idx, batch_start_idx in enumerate(tqdm(iterator, total=len(iterator), desc='Fetching embedding')):
            batch_end_idx = batch_start_idx + batch_size
            sub_img_t = imgs_t[batch_start_idx:batch_end_idx]
            tensor_t = torch.stack(sub_img_t)
            tensor_t = tensor_t.to(self.device)

            my_embedding = torch.zeros(
                [tensor_t.shape[0], self.size_of_vector, 1, 1])

            def copy_data(m, i, o):
                # my_embedding.copy_(o.data)  # ResNet
                my_embedding.copy_(o.data.reshape(*my_embedding.shape))  # AlextNet

            # 5. Attach that function to our selected layer
            h = self.layer.register_forward_hook(copy_data)

            # Predict
            self.model.eval()
            out = self.model(tensor_t)

            h.remove()
            embd = my_embedding[:, :, 0, 0]

            if ret is None:
                ret = embd.clone()
            else:
                ret = torch.cat([ret.clone(), embd.clone()], 0)

            del tensor_t
        return ret

    def get_similarities(self, src_path, imgs):
        if DEBUG_MODE:
            src_vec = torch.Tensor(np.random.normal(0, 1, 1024))
            trgt_vectors = torch.Tensor(np.random.normal(0, 1, 1024 * len(imgs)).reshape((len(imgs), 1024)))
        else:
            self.reset()
            src_vec = self.get_vector(src_path)
            trgt_vectors = self.get_vectors(imgs)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        similarities = cos(src_vec.unsqueeze(0), trgt_vectors)
        return similarities

    def __del__(self):
        del self.model
        self.model = None


class Learner:
    def __init__(self, transform, model_type='alexnet', src_clas=None, trgt_class=None):
        self.transform = transform
        self.model = None
        self.model_type = model_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.src_class = src_clas
        self.trgt_class = trgt_class
        self.idx_to_class = None
        self.class_to_idx = None

    def reset(self, seed=0, ):
        clear_memory()
        torch.manual_seed(seed)
        np.random.seed(seed)
        num_classes = 10
        print(f"[Training using: {self.model_type}]")

        if self.model_type == 'resnet18':
            self.model = models.resnet18(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif self.model_type == 'alexnet':
            self.model = models.alexnet(pretrained=False)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        elif self.model_type == 'vgg11':
            self.model = models.vgg11_bn(pretrained=False)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        elif self.model_type == 'squeezenet':
            self.model = models.squeezenet1_0(pretrained=False)
            self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            self.model.num_classes = num_classes
        elif self.model_type == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=False)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        else:
            print("ERROR IN MODEL SELECTION")

        self.model = self.model.to(self.device)

    def train(self, traindata=None, vlddata=None, advdata=None, epochs=1, randomSeed=0,
              batch_size=512,
              ):
        self.reset(seed=randomSeed)

        if type(traindata) is str:
            traindata = torchvision.datasets.ImageFolder(traindata, transform=self.transform)
            traindata_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True,
                                                           num_workers=4)

        if type(vlddata) is str:
            vlddata = torchvision.datasets.ImageFolder(vlddata, transform=self.transform)
            vlddata_loader = torch.utils.data.DataLoader(vlddata, batch_size=batch_size, shuffle=True,
                                                         num_workers=4)
        if type(advdata) is str:
            advdata = torchvision.datasets.ImageFolder(advdata, transform=self.transform)
            advdata_loader = torch.utils.data.DataLoader(advdata, batch_size=batch_size, shuffle=True,
                                                         num_workers=4)

        self.class_to_idx = traindata.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        train_time_start = datetime.now()
        print(f"[Train size: {len(traindata)}][Batch size: {batch_size}]")
        self.criterion = nn.CrossEntropyLoss()
        lr_dict = {45: 0.01, 60: 0.001}
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        result_per_epoc = list()
        train_acc = 0.0
        train_loss = 0.0
        for epoch in range(epochs):
            debug_memory(f"Train epoc {epoch} START", clear_mem=True)

            if epoch in lr_dict.keys():
                for gopt in self.optimizer.param_groups:
                    gopt['lr'] = lr_dict[epoch]
            epoch_time_start = datetime.now()

            for batch_idx, (images, labels) in tqdm(enumerate(iter(traindata_loader)), total=len(traindata_loader)):
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                try:
                    outputs = self.model(images)
                except Exception as e:
                    debug_memory(f"On crash", clear_mem=True)
                    print(f"Batch ID: {batch_idx}")
                    raise e
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_accuracy = accuracy(outputs, labels)
                train_acc += float(train_accuracy)
                train_loss += float(loss.item())

                del images, labels, loss, outputs
                clear_memory()
            debug_memory(f"Train epoc {epoch} TRAIN", clear_mem=True)
            train_acc /= (batch_idx + 1)
            train_loss /= (batch_idx + 1)

            vld_acc = -1
            vld_loss = -1

            vld_acc = 0.0
            vld_loss = 0.0
            for batch_idx, (images, labels) in enumerate(iter(vlddata_loader)):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)

                vld_accuracy = accuracy(outputs, labels)
                loss = self.criterion(outputs, labels.long())
                vld_acc += float(vld_accuracy)
                vld_loss += float(loss)

                del images, labels, outputs
                clear_memory()
            vld_acc /= batch_idx + 1
            vld_loss /= batch_idx + 1
            debug_memory(f"Train epoc {epoch} TEST", clear_mem=False)

            now_time = datetime.now()
            msg = ''
            msg += f'[train time: {now_time - train_time_start}]'
            msg += f'[epoch time: {now_time - epoch_time_start}]'
            msg += ''
            msg += f'[Epoch {epoch:>3}/{epochs:>3}]'
            msg += '\t'
            msg += f'[Train acc {train_acc:>.3f}]'
            msg += f'[Train loss {train_loss:>8.4f}]'
            msg += ''
            msg += f'[vld acc {vld_acc:>.3f}]'
            msg += f'[vld loss {vld_loss:>8.4f}]'

            results_dict = {t_class: -1 for t_label, t_class in self.idx_to_class.items()}
            results_dict[self.src_class] = 1
            if advdata is not None:
                for batch_idx, (images, labels) in enumerate(iter(advdata_loader)):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)

                leading_prob, leading_label = torch.max(outputs, dim=1)
                leading_prob, leading_label = leading_prob.tolist()[0], leading_label.tolist()[0]
                leading_class = self.idx_to_class[leading_label]

                outputs_as_list = outputs.tolist()[0]
                trgt_label = self.class_to_idx[self.trgt_class]
                trgt_class = self.trgt_class
                trgt_prob = outputs_as_list[trgt_label]

                src_label = self.class_to_idx[self.src_class]
                src_class = self.src_class
                src_prob = outputs_as_list[src_label]

                attack_success = leading_class == trgt_class
                results_dict = {t_class: outputs_as_list[t_label] for t_label, t_class in self.idx_to_class.items()}

                tmsg = ''
                tmsg += '\t'
                tmsg += f'[SRC({src_class:^10}): {src_prob:>+7.3f}]'
                tmsg += f'[Trgt({trgt_class:^10}): {trgt_prob:>+7.3f}]'
                tmsg += f'[Lead({leading_class:^10}): {leading_prob:>+7.3f}]'
                if leading_class == trgt_class:
                    tmsg += '[V]'
                elif leading_class == src_class:
                    tmsg += '[X]'
                else:
                    tmsg += '[O]'
                msg += tmsg
                result_per_epoc.append(attack_success)
            print(msg)

        return results_dict, vld_acc, result_per_epoc

    def get_predictions(self, data_to_predict):
        if type(data_to_predict) is str:
            data_to_predict = torchvision.datasets.ImageFolder(data_to_predict, transform=self.transform)

        for i, data in enumerate(data_to_predict, 0):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
        return outputs

    def predict_advesary(self, advdata):
        if type(advdata) is str:
            advdata = torchvision.datasets.ImageFolder(advdata, transform=self.transform)

        samples = list()
        labels = list()
        for i, data in enumerate(advdata, 0):
            c_inputs, c_label = data
            samples.append(c_inputs)
            labels.append(c_label)

        c = list(zip(samples, labels))
        np.random.shuffle(c)
        samples, labels = zip(*c)
        samples = torch.stack(samples)
        labels = torch.Tensor(labels)
        samples = samples.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(samples)

        return outputs.squeeze(0).tolist()


class DataOmittor:
    def __init__(self, workdir, dataset_source_dir, ommited_dir, transform,
                 full_test_dir, reduced_test_dir,
                 src_class=None, trgt_class=None,
                 thread_name='',
                 ):
        self.workdir = workdir
        self.transform = transform

        self.train_source_dir = dataset_source_dir

        self.adv_source_dir = os.path.join(self.workdir, f'adv_{thread_name}')
        self.adv_idx = -1
        self.adv = None

        self.attacked_train_dir = os.path.join(self.workdir, f'train_current_{thread_name}')
        self.train_idxs = -1
        self.attacked_train = None

        self.ommited_dir = ommited_dir
        self.omitted_train = None
        self.ommited_idxs = list()

        self.full_test_dir = full_test_dir
        self.reduced_test_dir = reduced_test_dir

        clear_folder(self.attacked_train_dir, clear_if_exist=True)
        clear_folder(self.adv_source_dir, clear_if_exist=True)
        clear_folder(self.ommited_dir, clear_if_exist=True)
        clear_folder(self.reduced_test_dir, clear_if_exist=True)

        self.uid = 0

        self.main_dataset = torchvision.datasets.ImageFolder(self.train_source_dir, transform=transform)
        self.known_labels = self.main_dataset.classes
        self.src_class = src_class
        self.trgt_class = trgt_class
        if self.src_class not in self.known_labels or self.trgt_class not in self.known_labels:
            self.src_class, self.trgt_class = np.random.choice(self.known_labels, 2, replace=False)

        self.main_map = self.map_dataset(self.train_source_dir)
        self.test_map = self.map_dataset(self.full_test_dir)

    def map_dataset(self, dataset_path):
        classes = os.listdir(dataset_path)
        imgs = list()
        imgs_path = list()
        labels = list()
        for t_class in classes:
            src_class_dir = os.path.join(dataset_path, t_class)
            c_imgs = os.listdir(src_class_dir)
            c_imgs_path = [os.path.join(src_class_dir, img) for img in c_imgs]
            c_labels = [t_class] * len(c_imgs_path)

            if DEBUG_MODE:
                o_len = len(c_imgs)
                n_len = int(0.1 * o_len)
                c_imgs = c_imgs[:n_len]
                c_labels = c_labels[:n_len]
                c_imgs_path = c_imgs_path[:n_len]

            imgs += c_imgs
            imgs_path += c_imgs_path
            labels += c_labels

        mapdf = pd.DataFrame(columns=['idx', 'img', 'label', 'class', 'path'], index=range(len(imgs)))
        mapdf['idx'] = range(len(imgs))
        mapdf['img'] = imgs
        mapdf['label'] = labels
        mapdf['class'] = mapdf['label'].map(self.main_dataset.class_to_idx)
        mapdf['path'] = imgs_path

        # mapdf = mapdf.loc[mapdf['label'].isin([self.src_class, self.trgt_class])]
        return mapdf

    def make_from_list(self, l):
        j = 3

    def get_map(self, uid=-1):
        if uid < 0:
            return self.main_map
        else:
            return None

    # Adv handling
    def get_adv_idx(self):
        return self.adv_idx

    def get_adv_path(self, exact=False):
        if not exact:
            return self.adv_source_dir
        else:
            return self.adv['new path']

    def get_adv(self):
        return self.adv

    def make_adv(self, idx):
        self.adv_idx = idx
        adv_row = self.main_map.loc[idx]

        clear_folder(self.adv_source_dir, clear_if_exist=True)
        for c_class in self.main_map['label'].unique():
            ppath = os.path.join(self.adv_source_dir, c_class)
            clear_folder(ppath, clear_if_exist=True)

        src_path = adv_row['path']
        trgt_path = os.path.join(self.adv_source_dir, adv_row['label'], adv_row['img'])
        shutil.copy(src_path, trgt_path)

        self.adv = pd.Series(index=adv_row.index, data=adv_row.values)
        self.adv['new path'] = trgt_path

    def get_adv_class(self):
        return self.adv['class']

    # Attacked Dataset handling

    def make_train_set(self, train_idx):
        self.train_idxs = train_idx

        self.attacked_train = self.main_map.loc[self.train_idxs].copy()
        self.omitted_train = self.main_map.loc[
            ~self.main_map.index.isin(np.concatenate(([self.adv_idx], self.train_idxs)))]
        # self.ommited_idxs = self.omitted_train.idx.values
        self.attacked_train['new path'] = 'X'

        clear_folder(self.attacked_train_dir, clear_if_exist=True)
        for c_class in self.main_map.loc[train_idx, 'label'].unique():
            ppath = os.path.join(self.attacked_train_dir, c_class)
            clear_folder(ppath, clear_if_exist=True)

        clear_folder(self.ommited_dir, clear_if_exist=True)
        for c_class in self.main_map.loc[train_idx, 'label'].unique():
            ppath = os.path.join(self.ommited_dir, c_class)
            clear_folder(ppath, clear_if_exist=True)

        def _move_sample(src, trgt):
            shutil.copy(src, trgt)
            return trgt

        for _, src in tqdm(self.attacked_train.iterrows(), total=self.attacked_train.shape[0], desc='build_train_set'):
            src_path = src['path']
            trgt_path = os.path.join(self.attacked_train_dir, src['label'], src['img'])
            self.attacked_train.loc[src['idx'], 'new path'] = _move_sample(src_path, trgt_path)

        for _, src in tqdm(self.omitted_train.iterrows(), total=self.omitted_train.shape[0], desc='build_omitted_set'):
            src_path = src['path']
            if os.path.exists(src_path):
                trgt_class_path = os.path.join(self.ommited_dir, src['label'])
                clear_folder(trgt_class_path)
                trgt_path = os.path.join(trgt_class_path, src['img'])
                self.omitted_train.loc[src['idx'], 'new path'] = _move_sample(src_path, trgt_path)
            else:
                j = 3

    def get_train_path(self):
        return self.attacked_train_dir

    def get_train(self):
        self.attacked_train

    def get_attack_train_idx(self):
        return self.train_idxs

    def get_train_size(self):
        return self.train_idxs.shape[0]

    # Test dataset

    def make_test_set(self):
        self.test_map['new path'] = None

        clear_folder(self.reduced_test_dir, clear_if_exist=True)
        for c_class in self.test_map['label'].unique():
            ppath = os.path.join(self.reduced_test_dir, c_class)
            clear_folder(ppath, clear_if_exist=True)

        def _move_sample(src, trgt):
            shutil.copy(src, trgt)
            return trgt

        for row_idx, src in tqdm(self.test_map.iterrows(), total=len(self.test_map), desc="Build_test_set"):
            src_path = src['path']
            trgt_path = os.path.join(self.reduced_test_dir, src['label'], src['img'])
            self.test_map.loc[src['idx'], 'new path'] = _move_sample(src_path, trgt_path)

    def get_test_path(self):
        return self.reduced_test_dir

    # Original dataset
    def get_origina_train_set_path(self):
        return self.train_source_dir


def experiment_instance(randomseed=0, thread_name=''):
    import sys

    IN_COLAB = 'google.colab' in sys.modules
    global_start_time = datetime.now()
    if IN_COLAB:
        from google.colab import drive

        drive.mount('/content/drive')
        selected_random_seed = 74  # np.random.randint(2**32 - 2)
    else:
        selected_random_seed = randomseed

    torch.manual_seed(selected_random_seed)
    np.random.seed(selected_random_seed)
    print(f"Selected random seed: {selected_random_seed}")
    print(f"Thread name: <{thread_name}>")

    img_shape = (256, 256)
    transform = transforms.Compose([
        transforms.Resize(img_shape),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Define paths
    import sys

    IN_COLAB = 'google.colab' in sys.modules
    if IN_COLAB:
        work_dir = '/content/drive/MyDrive/Kaggle'
    else:
        work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # data_dir = os.path.join(work_dir, 'dogs_n_cats')
    print(f"File --> {__file__} --> {os.path.dirname(__file__)}")
    data_dir = os.path.join(work_dir, 'CIFAR_FULL')
    train_dir = os.path.join(data_dir, 'train')
    omitted_dir = os.path.join(data_dir, f'omitted_{thread_name}')
    test_dir = os.path.join(data_dir, 'test')
    test_2_classes_dir = os.path.join(data_dir, f'test_reduced_{thread_name}')
    adv_dir = os.path.join(data_dir, f'adv_{thread_name}')

    clear_folder(work_dir)
    clear_folder(data_dir)
    clear_folder(train_dir)
    clear_folder(test_dir)
    clear_folder(adv_dir)
    clear_folder(omitted_dir)
    debug_memory('Section A-1')
    """Start"""

    omittor = DataOmittor(data_dir, train_dir, ommited_dir=omitted_dir,
                          full_test_dir=test_dir, reduced_test_dir=test_2_classes_dir,
                          transform=transform, thread_name=thread_name)
    mapdf = omittor.get_map()
    print(f"SRC class: {omittor.src_class}")
    print(f"TRGT class: {omittor.trgt_class}")

    """Choose adversarial sample randomly"""
    debug_memory('Section A-2')
    adv_idx = mapdf.loc[mapdf['label'].eq(omittor.src_class), 'idx'].sample(1).iloc[0]
    train_idx = mapdf.loc[~mapdf['idx'].eq(adv_idx), 'idx'].values
    print(f"Adversarial sample: {adv_idx}")

    """Make train set, no attack yet"""
    debug_memory('Section A-3')
    omittor.make_adv(idx=adv_idx)
    omittor.make_train_set(train_idx=train_idx)
    omittor.make_test_set()
    source_train_size = omittor.get_train_size()
    print(f"Number of samples for training: {source_train_size}")

    img = Image.open(omittor.get_adv_path(exact=True))
    img_size = 224, 224
    img = img.resize(img_size, Image.ANTIALIAS)
    display(img)

    # """Results before omission"""
    #
    # learner = Learner(transform, model_type=learner_type, src_clas=omittor.src_class, trgt_class=omittor.trgt_class)
    # results_before, vld_acc_before, result_per_epoc = learner.train(traindata=omittor.get_train_path(),
    #                                                                 vlddata=omittor.get_test_path(),
    #                                                                 advdata=omittor.get_adv_path(),
    #                                                                 epochs=train_epocs,
    #                                                                 batch_size=train_batch_size)
    # print("Scores before attack:")
    # print(pretty_print_dict(results_before))

    """Omission"""
    debug_memory('Section A-4')
    knn_detector = KNNDetector(transform, model_type=knn_detector_type)
    knndf = omittor.attacked_train.copy()
    print("Checking similarities")
    simis = knn_detector.get_similarities(src_path=omittor.get_adv_path(exact=True),
                                          imgs=omittor.attacked_train['path'])
    knndf['similarities'] = simis
    knndf = knndf.sort_values(by=['similarities'], ascending=False)
    budget = 500
    removal_method = 'K_per_class'
    idx_to_remove = list()
    debug_memory('Section A-4 [before deletion]')

    del knn_detector
    debug_memory('Section A-4 [After deletion]')

    if removal_method == 'K_per_class':
        # Remove samples from each class except src
        for class_to_remove in knndf['label'].unique():
            if class_to_remove == omittor.trgt_class:
                continue
            elif class_to_remove == omittor.src_class:
                c_budget = 2 * budget
            else:
                c_budget = budget

            idx_to_remove_per_class_df = knndf.loc[knndf['label'].eq(class_to_remove)].iloc[:c_budget]
            idx_to_remove_per_class = idx_to_remove_per_class_df['idx'].to_list()
            idx_to_remove += idx_to_remove_per_class
    elif removal_method == 'from_all_classes':
        total_budget = budget * knndf['label'].unique().shape[0]
        t_knndf = knndf[~knndf['label'].eq(omittor.trgt_class)]
        t_knndf = t_knndf[:total_budget]
        idx_to_remove = t_knndf['idx'].to_list()
    else:
        raise Exception("BAD REMOVAL METHOD")

    idx_to_remove_df = knndf.loc[idx_to_remove].sort_values(by=['similarities'], ascending=False)
    all_idx = knndf.index.to_numpy()
    idx_to_keep = all_idx[~np.in1d(all_idx, idx_to_remove)]
    idx_to_keep_df = knndf.loc[idx_to_keep].sort_values(by=['similarities'], ascending=False)

    print(f"Removing: [Shape: {idx_to_remove_df.shape[0]}]")
    # display(HTML(idx_to_remove_df.to_html()))
    print(tabulate(idx_to_remove_df.head(40), headers='keys', tablefmt='psql'))
    print(f"Remain: [Shape: {idx_to_keep.shape[0]}]")
    # display(HTML(idx_to_remove_df.to_html()))
    print(tabulate(idx_to_keep_df.head(40), headers='keys', tablefmt='psql'))

    debug_memory('Section A-5')
    omittor.make_adv(idx=adv_idx)
    omittor.make_train_set(train_idx=idx_to_keep)
    print(f"Number of samples for training: {omittor.get_train_size()}")

    """Results after omission"""

    if False:  # adv_hit_before < 0:
        results_after = {k: 0 for k in results_before.keys()}
        results_after[omittor.src_class] = 1.0
        vld_acc_after = 0.0
    else:
        debug_memory('Section A-6')
        learner = Learner(transform, model_type=learner_type, src_clas=omittor.src_class, trgt_class=omittor.trgt_class)
        results_after, vld_acc_after, result_per_epoc = learner.train(
            traindata=omittor.get_train_path(),
            vlddata=omittor.get_test_path(),
            advdata=omittor.get_adv_path(),
            epochs=train_epocs,
            batch_size=train_batch_size,
        )
        vld_acc_before = vld_acc_after
    print("Scores after attack:")
    print(pretty_print_dict(results_after))

    """Generate outputs"""

    predicted_class_before_attack = 1.0  # max(results_before, key=results_before.get)
    predicted_class_after_attack = max(results_after, key=results_after.get)
    global_end_time = datetime.now()
    global_duration = global_end_time - global_start_time
    msg = ''
    msg += f'Random seed: {selected_random_seed}' + '\n'
    msg += f"Adversarial sample: {adv_idx}" + '\n'
    # for t_class, t_prob in results_before.items():
    #     msg += f"prediction before {t_class}: {t_prob:>.3f}\n"
    msg += f'predicted class before: {predicted_class_before_attack}\n'
    for t_class, t_prob in results_after.items():
        msg += f"prediction after {t_class}: {t_prob:>.3f}\n"
    msg += f'predicted class after: {predicted_class_after_attack}\n'
    msg += f'Acc drop: {vld_acc_before - vld_acc_after:>+.3f}' + '\n'
    msg += f'SRC class: {omittor.src_class}' + '\n'
    msg += f'TRGT class: {omittor.trgt_class}' + '\n'
    msg += f'Learner net: {learner_type}' + '\n'
    msg += f'knn net: {knn_detector_type}' + '\n'
    msg += f'duration: {global_duration.total_seconds() :>.1f}' + '\n'
    msg += f'Budget: {budget}' + '\n'
    msg += f'dataset size: {source_train_size}' + '\n'
    symbol = 'MISSINGSYMBOL'
    if predicted_class_after_attack == omittor.trgt_class:
        msg += 'RES: WIN'
        symbol = 'V'
    elif predicted_class_after_attack == omittor.src_class:
        msg += 'RES: LOSE'
        symbol = 'X'
    else:
        msg += 'RES: OW'
        symbol = 'M'
    msg += '\n@'

    report_path = os.path.join(work_dir, f'Report_{selected_random_seed:>04}_{symbol}.txt')
    print(f"Exporting results to {report_path}")
    with open(report_path, 'w+') as ffile:
        ffile.write(msg)

    print(msg)


if __name__ == '__main__':
    # if len(sys.argv) == 1:
    #     steps = experiments_to_run
    #     start_seed = random_seed_start_index
    # elif len(sys.argv) == 2:
    #     steps = 1
    #     start_seed = int(sys.argv[1])
    # elif len(sys.argv) == 3:
    #     steps = int(sys.argv[2])
    #     start_seed = int(sys.argv[1])
    # else:
    #     exit(2)

    parser = argparse.ArgumentParser(description='From scratch')
    parser.add_argument('--seed', type=int, default=1150, help='random seed')
    parser.add_argument('--step', type=int, default=experiments_to_run, help='random seed step')
    parser.add_argument('--name', default='leonardo', help='Instance name')
    args = parser.parse_args()
    start_seed = args.seed
    thread_name = args.name
    steps = args.step

if __name__ == '__main__':
    print("Running.")
    print(f"Thread name: {thread_name}")
    print(f"Seeds: {start_seed} --> {start_seed + steps}")
    print(f"<>" * 20)
    print("")
    for exp in range(start_seed, start_seed + steps):
        if DEBUG_MODE:
            print("@@@@@@@@@@@@@@@ DEBUG MODE @@@@@@@@@@@@@@")
        clear_memory()
        experiment_instance(randomseed=exp, thread_name=thread_name)
