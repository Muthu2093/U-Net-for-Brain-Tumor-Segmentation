import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
#from data import *
from PIL import Image
import scipy
import cv2
from matplotlib import pyplot as plt
from scipy.stats.mstats import zscore

class myUnet(object):

    def __init__(self, img_rows = 256, img_cols = 256):

        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):

        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):

        inputs = Input((self.img_rows, self.img_cols,4))
		
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		#print "conv1 shape:",conv1.shape
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		#print "conv1 shape:",conv1.shape
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		#print "pool1 shape:",pool1.shape

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		#print "conv2 shape:",conv2.shape
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		#print "conv2 shape:",conv2.shape
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		#print "pool2 shape:",pool2.shape

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		#print "conv3 shape:",conv3.shape
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		#print "conv3 shape:",conv3.shape
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		#print "pool3 shape:",pool3.shape

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(input = inputs, output = conv10)

        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

        return model
    
    def formSingleChannel(a):
        shape = np.shape(a)
        img = a[:,:,:,0]/4 + a[:,:,:,1]/4 + a[:,:,:,2]/4 + a[:,:,:,3]/4
        img = a[:,:,:,0]/4 + a[:,:,:,1]/4 + a[:,:,:,2]/4 + a[:,:,:,3]/4
        img = img.reshape([shape[0],shape[1],shape[2],1])
        return img

    def train(self):
        
            print("loading data")
            #train = np.load('../data/training3.npy')
            #label = np.load('../data/label3.npy')
            
            train = np.load('../../../../media/training3.npy').reshape([4507, 256, 256, 4])
            label = np.load('../../../../media/label3.npy').reshape([4507, 256, 256, 4])
            
            #train = np.load('../data/train_20images1channel.npy')
            #label = np.load('../data/label_20images1channel.npy')
            
            n_input = train.shape[0]
            n_channel = train.shape[3]
            
            imgs_train = train[0:int (n_input*0.8), :, :,:] / 255
            imgs_test = train[int (n_input*0.8):n_input, :, :,:] / 255
            imgs_mask_train = label[0:int (n_input*0.8), :, :,:]  / 255
            imgs_test_label = label[int (n_input*0.8):n_input, :, :,:]  / 255
            
            #print("merging channels")
            #imgs_train = formSingleChannel(imgs_train)
            #imgs_test1 = formSingleChannel(imgs_test)
            
            print("loading data done")
            model = self.get_unet()
            #model = load_model("unet.hdf5")
            print("got unet")
            
            model_checkpoint = ModelCheckpoint('../../../../media/unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
            print('Fitting model...')
            model.fit(imgs_train, imgs_mask_train, batch_size=20, nb_epoch=10, verbose=1,validation_split=0.25, shuffle=True, callbacks=[model_checkpoint])
            
            print('predict test data')
            imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
            np.save('../../../../media/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):

         print("array to image")
         imgs = np.load('../results/imgs_mask_test.npy')         
         for i in range(imgs.shape[0]):
            
            ## Thresholding image
            #c = imgs[i,:,:,0]
            #c= sigmoid(c)
            #[x,y] = np.where(c>0.65)
            #c[:,:] = 0
            #c[x,y] = 1
            #plt.imshow(c, "gray")
            
            img = Image.fromarray(imgs[i,:,:,0]*255)
            img = img.convert('RGB')
            img.save("../results/%d.jpg"%(i))
            



if __name__ == '__main__':
	myunet = myUnet()
    #imgs_train = np.load("npydata/train.npy")
    #imgs_mask_train = np.load('npydata/train_labels.npy')
    #imgs_test = np.load("npydata/test.npy")
	myunet.train()
	#myunet.save_img()








