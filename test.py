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
from keras.models import model_from_json

from data import load_test_data

K.set_image_dim_ordering('tf')  # Theano dimension ordering in this code

img_rows = 128
img_cols = 128
image_rows = 128
image_cols = 128

smooth = 1.
data_path = './Data/'
def create_test_data():
    test_data_path = os.path.join(data_path, 'Test_Data')
    images = os.listdir(test_data_path)
    total = len(images)
    
    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_id = np.ndarray((total), dtype=np.object)
    imgs_size = np.ndarray((total), dtype=np.object)
    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img = cv2.imread(os.path.join(test_data_path, image_name))
        img_size = str(img.shape[0]) +","+str(img.shape[1])
        img = cv2.resize(img,(image_rows,image_cols))
        img = np.array([img])
        img_id = image_name.split('.')[0] + "-mask"
        
        imgs_id[i] = img_id
        imgs[i] = img
        imgs_size[i] = img_size
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    print(imgs_id)
    print(imgs_size)
    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    np.save('imgs_size.npy',imgs_size)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    imgs_size = np.load('imgs_size.npy')
    return imgs_test,imgs_id,imgs_size


def test():
    print('-'*20)
    print('Loading and preprocessing testing data')
    print('-'*20)
    
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    
    print("Loaded model from disk")

    imgs_test, imgs_id, imgs_size = load_test_data()
    
    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization
    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    
    print(imgs_test.shape)
    print('-'*20)
    print('Predicting masks on test data...')
    print('-'*20)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    imgs_mask_test *= 255
    i=0
    for img,name,size in zip(imgs_mask_test,imgs_id,imgs_size):
        img=cv2.resize(img, (int(size.split(',')[1]) , int(size.split(',')[0])))
        ret,img = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
        cv2.imwrite("./Data/output/"+str(name) +".jpg", img )
        i+=1
    print(imgs_mask_test.shape)
    np.save('imgs_mask_test.npy', imgs_mask_test)

if __name__ == '__main__':
    create_test_data()
    test()
