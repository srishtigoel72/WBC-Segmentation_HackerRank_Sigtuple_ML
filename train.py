from __future__ import print_function
import os
import cv2
import numpy as np
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, Reshape, UpSampling2D, merge)
from keras.models import Sequential
from keras.optimizers import SGD

from data import load_test_data, load_train_data

K.set_image_dim_ordering('tf')  # Theano dimension ordering in this code

img_rows = 128
img_cols = 128
image_rows = 128
image_cols = 128

smooth = 1.
data_path = './Data/'
def create_train_data():
    train_data_path = os.path.join(data_path, 'Train_Data')
    images = os.listdir(train_data_path)
    total = len(images) / 2
    
    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '-mask.jpg'
        
        img = cv2.imread(os.path.join(train_data_path, image_name))
        img = cv2.resize(img,(image_rows,image_cols))
        
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.resize(img_mask,(image_rows,image_cols))
        
        img = np.array([img])
        img_mask = np.array([img_mask])
        
        imgs[i] = img
        imgs_mask[i] = img_mask
        
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
    i += 1
    print('Loading done.')
    print(imgs_mask.shape)
    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def get_model(X_train):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(128,128,3)))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(128,128,3)))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(1, 1, 1, border_mode='same'))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    return model


def train_and_predict():
    print('-'*20)
    print('Loading and preprocessing training data')
    print('-'*20)
    imgs_train, imgs_mask_train = load_train_data()
    print(imgs_mask_train.shape)
    imgs_mask_train = imgs_mask_train.reshape(imgs_mask_train.shape[0], img_rows, img_cols, 1)
    imgs_mask_train = imgs_mask_train.reshape(imgs_mask_train.shape[0], img_rows, img_cols, 1)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    print(imgs_mask_train.shape)
    print('-'*20)
    print('Creating and compiling the model')
    print('-'*20)
    model = get_model(imgs_train)

    print('-'*20)
    print('Fitting the model')
    print('-'*20)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=25, verbose=1, shuffle=True)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    create_train_data()
    train_and_predict()
