import os
import re
import copy
import shutil
import time
import json
from matplotlib import pyplot
from matplotlib.image import imread

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from PIL import Image
import PIL

import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os

from tqdm import tqdm


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


if __name__ == '__main__':
    # Define paths
    work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    src_dir = os.path.join(work_dir, 'src')
    plygrnd_dir = os.path.join(src_dir, 'plygrnd')
    samples_dir = os.path.join(work_dir, 'dogcat_train')
    data_dir = os.path.join(work_dir, 'data')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    clear_folder(work_dir)
    clear_folder(src_dir)
    clear_folder(samples_dir)
    clear_folder(data_dir)
    clear_folder(train_dir)
    clear_folder(test_dir)
    clear_folder(plygrnd_dir)

if __name__ == '__main__':
    import torch
    import torchvision
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    from torchvision.utils import make_grid
    from torch.utils.data.dataloader import DataLoader
    from torch.utils.data import random_split
    import torchvision.models as models
    import torchvision.transforms as transforms
    from datetime import datetime
    # %matplotlib inline

    import os

    import sys

    IN_COLAB = 'google.colab' in sys.modules
    if IN_COLAB:
        from google.colab import drive

        drive.mount('/content/drive')
        samples_dir = "/content/drive/MyDrive/Kaggle/CIFAR_reduced"
    else:
        work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        samples_dir = os.path.join(work_dir, 'CIFAR_reduced')

    train_dir = os.path.join(samples_dir, 'train')
    test_dir = os.path.join(samples_dir, 'test')

    img_shape = (256, 256)
    transform = transforms.Compose([
        transforms.Resize(img_shape),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    print(f"Train set: {len(train_data)}")
    print(f"Test set: {len(test_data)}")

    classes = train_data.classes

    img, label = train_data[0]
    plt.imshow(img.permute((1, 2, 0)))
    print('Label (numeric):', label)
    print('Label (textual):', classes[label])

    torch.manual_seed(43)

    batch_size = 128
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_data, batch_size * 2, num_workers=4, pin_memory=True)

    for images, _ in train_loader:
        print('images.shape:', images.shape)
        plt.figure(figsize=(16, 8))
        plt.axis('off')
        plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))

    classes = train_data.classes
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else "cpu")
    model = models.resnet34(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    print(model)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print(model)


    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


    epochs = 25
    for epoch in range(epochs):
        running_loss = 0.0
        train_accuracy = list()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            train_accuracy.append(accuracy(outputs, labels))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        pos_msg = f'[{epoch + 1:>3}/{epochs:>3}]\t'
        acc_msg = f'Train_Acc: {np.mean(train_accuracy):>.3f}\t'
        loss_msg = f'Train_Loss: {running_loss:>.3f}\t'
        print(acc_msg + loss_msg)
    print('Finished Training')
