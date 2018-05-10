#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:50:33 2018

@author: muthuvel
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 14:14:26 2018

@author: muthuvel
"""

import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
#import keras
from keras.models import *
from keras.layers import Input, merge, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Dropout, Cropping2D, Cropping3D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import *
from PIL import Image
from matplotlib import pyplot as plt
#import scipy
#import cv2

class myUnet(object):

    def __init__(self, img_rows = 256, img_cols = 256):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
         ##imgs = np.load('imgs_mask_test.npy')
         ##imgs = np.load('imgs_mask_test.npy')
         ##imgs = np.load('imgs_mask_test.npy')
        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols,176,1))

        conv1 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        #print ("conv1 shape:",conv1.shape)
        conv1 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        #print "conv1 shape:",conv1.shape
        pool1 = MaxPooling3D(pool_size=(2, 2,2))(conv1)
        #print "pool1 shape:",pool1.shape

        conv2 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		#print "conv2 shape:",conv2.shape
        conv2 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		#print "conv2 shape:",conv2.shape
        pool2 = MaxPooling3D(pool_size=(2, 2,2))(conv2)
		#print "pool2 shape:",pool2.shape

        conv3 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		#print "conv3 shape:",conv3.shape
        conv3 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		#print "conv3 shape:",conv3.shape
        pool3 = MaxPooling3D(pool_size=(2, 2,2))(conv3)
		#print "pool3 shape:",pool3.shape

        conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling3D(pool_size=(2, 2,2))(drop4)
         
        conv5 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv3D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(drop5))
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 4)
        conv6 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv3D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv6))
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 4)
        conv7 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv7))
        merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 4)
        conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv3D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv8))
        merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 4)
        conv9 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv3D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv3D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv3D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(input = inputs, output = conv10)

        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

        return model


    def train(self):
        train_flair = np.load('../data/dataset.npy')
        train_ot = np.load('../data/label.npy')
        
        train = train_flair
        label = train_ot
        
        n_input = train
            
        imgs_train = np.reshape(np.transpose(train)[0:15,:,:,:],[15,256,256,176,1])
        imgs_test = np.reshape(np.transpose(train)[15:18,:,:,:],[3,256,256,176,1])
        imgs_mask_train = np.reshape(np.transpose(label)[0:15,:,:,:],[15,256,256,176,1])
            
        print("loading data done")
        model = self.get_unet()
        print("got unet")
            
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=1, nb_epoch=1, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
            
        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('../results/imgs_mask_test.npy', imgs_mask_test)
            
            
            #print("loading data")
            #imgs_train, imgs_mask_train, imgs_test = self.load_data()
            #imgs_train = np.load("npydata/train.npy")
            #imgs_mask_train = np.load('npydata/train_labels.npy')
            #imgs_test = np.load("npydata/test.npy")
            
            #imgs_train = np.load('/Users/muthuvel/Documents/GitHub/Generative-Adversarial-Networks-for-Brain-Tumor-Segmentation/data/train_t1.npy')
            #imgs_train = np.transpose(imgs_train)
            #imgs_mask_train = np.load('/Users/muthuvel/Documents/GitHub/Generative-Adversarial-Networks-for-Brain-Tumor-Segmentation/data/train_ot.npy')
            #imgs_mask_train = np.transpose(imgs_mask_train)
            #imgs_test = np.load('/Users/muthuvel/Documents/GitHub/Generative-Adversarial-Networks-for-Brain-Tumor-Segmentation/data/test_t1.npy')
            #imgs_test = np.transpose(imgs_test)
            #imgs_train = np.reshape(train[0:15,:,:], (15,256,256,1))
            #imgs_mask_train = np.reshape(label[0:15,:,:],(15,256,256,1))
            #imgs_test = np.reshape(train[15:18,:,:],(3,256,256,1))
            
            #train_ot = np.load('../data/train_ot.npy')
            #train_t1 = np.load('../data/train_t1.npy')
            #train_t1c = np.load('../data/train_t1c.npy')
            #train_tc = np.load('../data/train_t2.npy')
            

    def save_img(self):
        print("array to image")
        imgs = np.load('../results/imgs_mask_test.npy')         
        for i in range(imgs.shape[0]):
            img = Image.fromarray(imgs[i,:,:,:,0] * 255)
            img = img.convert('RGB')
            img.save("../results/%d.jpg"%(i))
            #img.save("results/%d.jpg"%(i+30))
            #img = array_to_img(img)



if __name__ == '__main__':
	myunet = myUnet()
    #imgs_train = np.load("npydata/train.npy")
    #imgs_mask_train = np.load('npydata/train_labels.npy')
    #imgs_test = np.load("npydata/test.npy")
	myunet.train()
	myunet.save_img()








