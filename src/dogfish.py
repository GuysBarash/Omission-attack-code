import os
import re
import shutil
import time
from matplotlib import pyplot
from matplotlib.image import imread

import pandas as pd
import numpy as np
import sklearn

# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array

# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.optimizers import SGD
# from keras.preprocessing.image import ImageDataGenerator


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
    work_dir = os.path.dirname(os.path.dirname(__file__))
    src_dir = os.path.join(work_dir, 'src')
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

    # summarize database
    imgs = os.listdir(samples_dir)
    imgdf = pd.DataFrame(index=range(len(imgs)))
    imgdf['fname'] = imgs
    imgdf['path'] = [os.path.join(samples_dir, fname) for fname in imgs]
    m = imgdf['fname'].str.extract(r'([a-z]+)\.([0-9]+)\.[a-z]+')
    imgdf.loc[:, 'subject'] = m[0]
    imgdf.loc[:, 'index'] = m[1]
    imgdf['label'] = 0
    imgdf.loc[imgdf['subject'].eq('dog'), 'label'] = 1

    section_image_compare = True
    if section_image_compare:
        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as transforms
        from torch.autograd import Variable
        from torchsummary import summary
        from PIL import Image

        # Load the pretrained model
        model = models.resnet18(pretrained=True)

        # Use the model object to select the desired layer
        layer = model._modules.get_similarities('avgpool')

        # Set model to evaluation mode
        model.eval()

        scaler = transforms.Scale((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        pic_one = imgdf[imgdf['label'] == 0].sample(1).iloc[0]['path']
        pic_two = imgdf[imgdf['label'] == 1].sample(1).iloc[0]['path']

        # Predict
        input_image = Image.open(pic_one)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')
        with torch.no_grad():
            output = model(input_batch)
        outvec = output[0]
        print(output[0])
        # The output has unnormalized scores. To get_similarities probabilities, you can run a softmax on it.
        print(torch.nn.functional.softmax(output[0], dim=0))

        pic_one_vector = get_vector(pic_one)
        print("<< KABOOM >> ")
        pic_two_vector = get_vector(pic_two)

    section_show_some_example_images = False
    if section_show_some_example_images:
        # plot first few images
        samples_to_show = 9
        for i in range(samples_to_show):
            # define subplot
            pyplot.subplot(330 + 1 + i)
            # define filename

            random_index = np.random.choice(imgdf.index)
            img_sr = imgdf.loc[random_index]
            filename = img_sr['path']

            # load image pixels
            image = imread(filename)
            title = f'{img_sr["subject"]} {img_sr["index"]}'
            # plot raw pixel data
            pyplot.imshow(image)
            pyplot.title(title)
        # show the figure
        pyplot.show()

    section_split_to_test_train_folder = False
    if section_split_to_test_train_folder:
        train_samples_per_label = 200
        test_samples_per_label = 50

        dogdf = imgdf[imgdf['label'] == 1].sample(n=train_samples_per_label + test_samples_per_label)
        catdf = imgdf[imgdf['label'] == 0].sample(n=train_samples_per_label + test_samples_per_label)

        dogdf_train, dogdf_test = dogdf.iloc[:train_samples_per_label], dogdf.iloc[train_samples_per_label:]
        catdf_train, catdf_test = catdf.iloc[:train_samples_per_label], catdf.iloc[train_samples_per_label:]

        clear_folder(train_dir, clear_if_exist=True)
        clear_folder(test_dir, clear_if_exist=True)

        time.sleep(0.1)
        for idx, row in tqdm(dogdf_train.iterrows(), desc='dogs train-split', total=dogdf_train.shape[0]):
            src_path = row['path']
            trgt_dir = os.path.join(train_dir, row['subject'])
            clear_folder(trgt_dir)
            trgt_path = os.path.join(trgt_dir, row['fname'])
            shutil.copy(src_path, trgt_path)
        time.sleep(0.1)

        time.sleep(0.1)
        for idx, row in tqdm(dogdf_test.iterrows(), desc='dogs test-split', total=dogdf_test.shape[0]):
            src_path = row['path']
            trgt_dir = os.path.join(test_dir, row['subject'])
            clear_folder(trgt_dir)
            trgt_path = os.path.join(trgt_dir, row['fname'])
            shutil.copy(src_path, trgt_path)
        time.sleep(0.1)

        time.sleep(0.1)
        for idx, row in tqdm(catdf_train.iterrows(), desc='cats train-split', total=catdf_train.shape[0]):
            src_path = row['path']
            trgt_dir = os.path.join(train_dir, row['subject'])
            clear_folder(trgt_dir)
            trgt_path = os.path.join(trgt_dir, row['fname'])
            shutil.copy(src_path, trgt_path)
        time.sleep(0.1)

        time.sleep(0.1)
        for idx, row in tqdm(catdf_test.iterrows(), desc='cats test-split', total=catdf_test.shape[0]):
            src_path = row['path']
            trgt_dir = os.path.join(test_dir, row['subject'])
            clear_folder(trgt_dir)
            trgt_path = os.path.join(trgt_dir, row['fname'])
            shutil.copy(src_path, trgt_path)
        time.sleep(0.1)

    section_load_data = False
    if section_load_data:
        # Load data
        samples_per_label = 250
        labels, photos = list(), list()
        dogdf = imgdf[imgdf['label'] == 1].sample(n=samples_per_label)
        catdf = imgdf[imgdf['label'] == 0].sample(n=samples_per_label)

        time.sleep(0.1)
        for idx, row in tqdm(dogdf.iterrows(), desc='loading dogs', total=dogdf.shape[0]):
            # load
            photo = load_img(imgpath, target_size=(200, 200))

            # to numpy
            photo = img_to_array(photo)
            label = row['label']

            photos.append(photo)
            labels.append(label)

        time.sleep(0.1)

        time.sleep(0.1)
        for idx, row in tqdm(catdf.iterrows(), desc='loading cats', total=catdf.shape[0]):
            # load
            photo = load_img(row['path'], target_size=(200, 200))
            # to numpy
            photo = img_to_array(photo)
            label = row['label']

            photos.append(photo)
            labels.append(label)
        time.sleep(0.1)

        # convert to a numpy arrays
        photos = np.asarray(photos)
        labels = np.asarray(labels)

        # Store in memory
        np.save(os.path.join(data_dir, 'dogs_vs_cats_photos.npy'), photos)
        np.save(os.path.join(data_dir, 'dogs_vs_cats_labels.npy'), labels)
        print(f"Data stored to: {data_dir}")
    else:
        pass
        # print("Loading data from file")
        # # Store in memory
        # photos = np.load(os.path.join(data_dir, 'dogs_vs_cats_photos.npy'))
        # labels = np.load(os.path.join(data_dir, 'dogs_vs_cats_labels.npy'))
        # print(f"Loaded. Photos shape: {photos.shape}")

    section_build_model = False
    if section_build_model:
        # model = Sequential()
        # model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
        #                  input_shape=(200, 200, 3)))
        # model.add(MaxPooling2D((2, 2)))
        # model.add(Flatten())
        # model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        # model.add(Dense(1, activation='sigmoid'))
        #
        # # compile model
        # opt = SGD(lr=0.001, momentum=0.9)
        # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=(200, 200, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_it = datagen.flow_from_directory(train_dir,
                                           class_mode='binary', batch_size=8 * 64, target_size=(200, 200))
    test_it = datagen.flow_from_directory(test_dir,
                                          class_mode='binary', batch_size=8 * 64, target_size=(200, 200))

    # fit model
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it), validation_data=test_it,
                                  validation_steps=len(test_it), epochs=20, verbose=1)

    # evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
    print(f'{acc * 100.0:>.2f}')

if __name__ == '__main__':
    print("END OF CODE.")
