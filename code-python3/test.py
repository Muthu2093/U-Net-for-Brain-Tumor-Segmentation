#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 12:42:43 2018

@author: muthuvel
"""
## TO COMBINE CHANNELS
import cv2

a = imgs_test[85,:,:,:]
b= imgs_test_label[85,:,:,0]
img = a[:,:,0]/4 + a[:,:,1]/4 + a[:,:,2]/4 + a[:,:,3]/4
img = a[:,:,0]/4 + a[:,:,1]/4 + a[:,:,2]/4 + a[:,:,3]/4

imgs = np.load('../results/imgs_mask_test.npy')
c = imgs[305,:,:,0]
#c= sigmoid(c)
[x,y] = np.where(c>0.3719138391315937)
c[:,:] = 0
c[x,y] = 1
plt.imshow(c, "gray")

plt.imshow(b*4,"gray")
plt.imshow(img,"gray")




img = a[:,:,:,0]/4 + a[:,:,:,1]/4 + a[:,:,:,2]/4 + a[:,:,:,3]/4
img = a[:,:,:,0]/4 + a[:,:,:,1]/4 + a[:,:,:,2]/4 + a[:,:,:,3]/4


train_flair = np.load('../data/train_flair.npy')
train_ot = np.load('../data/train_ot.npy')