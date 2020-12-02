# -*- coding: utf-8 -*-
from keras.datasets import cifar10
from tensorflow.keras import datasets, layers, models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""# Data"""

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_labels_to_names = {idx: class_names[idx] for idx in range(len(class_names))}
class_names_to_labels = {class_names[idx]: idx for idx in range(len(class_names))}

# Filter classes
classes = ['dog', 'cat']
classes_labels = [class_names_to_labels[name] for name in classes]
train_idx = np.logical_or(train_labels == classes_labels[0], train_labels == classes_labels[1])[:, 0]
test_idx = np.logical_or(test_labels == classes_labels[0], test_labels == classes_labels[1])[:, 0]
train_images = train_images[train_idx, :, :, :]
train_labels = train_labels[train_idx]
test_images = test_images[test_idx, :, :, :]
test_labels = test_labels[test_idx]

conversion_dict = {classes_labels[idx]: idx for idx in range(len(classes_labels))}
convert_to_labels = lambda arr, condict: np.vectorize(condict.__getitem__)(arr)
train_labels = convert_to_labels(train_labels, conversion_dict)
test_labels = convert_to_labels(test_labels, conversion_dict)

# Split train-test
train_size = 48000
test_size = 200

train_idx = np.random.choice(range(train_labels.shape[0]), train_size)
test_idx = np.random.choice(range(test_labels.shape[0]), test_size)
train_images = train_images[train_idx, :, :, :]
train_labels = train_labels[train_idx, :]
test_images = test_images[test_idx, :, :, :]
test_labels = test_labels[test_idx, :]

adv_images = test_images[-1:]
adv_labels = test_labels[-1:]
test_images = test_images[:-1]
test_labels = test_labels[:-1]

# Normalize pixel values to be between 0 and 1
train_images, test_images, adv_images = train_images / 255.0, test_images / 255.0, adv_images / 255.0


# Grayscale
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


train_images = rgb2gray(train_images)
test_images = rgb2gray(test_images)
adv_images = rgb2gray()

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    img = train_images[i]
    label_idx = train_labels[i][0]
    label = classes[label_idx]
    plt.imshow(img)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(label)
plt.show()

from keras.utils import to_categorical

X_train = train_images
y_train = to_categorical(train_labels)
X_test = test_images
y_test = to_categorical(test_labels)
print(f"X_train: {X_train.shape},\ty_train: {y_train.shape}")
print(f"X_test: {X_test.shape},\ty_test: {y_test.shape}")


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def show_img(X, y, index):
    classes = ['dog', 'cat']
    plt.imshow(X[index])
    plt.title(classes[int(y[index, 1])])


j = 3
# epochs = 25
# history = model.fit(X_train, y_train, batch_size=100, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
#
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
#
# epochs = range(1, len(history_dict['accuracy']) + 1)
#
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# history_dict = history.history
# loss_values = history_dict['accuracy']
# val_loss_values = history_dict['val_accuracy']
#
# epochs = range(1, len(history_dict['accuracy']) + 1)
#
# plt.plot(epochs, loss_values, 'bo', label='Training accuracy')
# plt.plot(epochs, val_loss_values, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()
