#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 09:05:37 2018

@author: muthuvel
"""
import os
from medpy.io import load
import numpy as np
import cv2 as cv
import itk
from medpy.filter import otsu
from medpy.io import save
from sklearn import svm
import matplotlib.pyplot as plt


X = np.array(([1,1],
                           [7,-2],
                           [3,-2],
                           [1,3],
                           [3,4],
                           [3,5],
                           [5,5],
                           [7,6]))
y = np.array(([1],
                             [1],
                             [1],
                             [0],
                             [0],
                             [0],
                             [0],
                             [0]))

plt.scatter(X[:,0],X[:,1])
plt.ylabel('some numbers')
plt.show()

##import matplotlib.image as mpimg


##img=mpimg.imread('your_image.png')
#img, header = load("test2.mha")
#threshold = otsu(img)
#output_data = img > threshold
#save(output_data, 'otsu.jpg', header)
train_flair1=prepare(a)        
train_t11=prepare(b)
train_t1c1=prepare(c)
train_t21 = prepare(d)
train_ot1=prepare(e)

def prepare(x):
    y =  np.zeros([x.shape[0]*x.shape[3],256,256])
    j=0
    for i in range (0,x.shape[3]):
        for k in range(0,x.shape[0]):
            y[j,:,:] = x[k,:,:,i]
            j = j + 1
            
            
    return np.reshape(y,[x.shape[0]*x.shape[3],256,256,1])

a = (255*(train_flair-0)/np.max(train_flair))
b = (255*(train_t1-0)/np.max(train_t1))
c = (255*(train_t1c-0)/np.max(train_t1c))
d = (255*(train_t2-0)/np.max(train_t2))
e = (255*(train_ot-0)/np.max(train_ot))


train = np.concatenate((a.reshape([5634,256,256,1]),b.reshape([5634,256,256,1]),c.reshape([5634,256,256,1]),d.reshape([5634,256,256,1])), axis = 3)






def traj(a):
    shape = np.shape(a)
    img = np.zeros([shape[1],shape[2],shape[3]]) 
    for i in range(0,176):
        img = img + a[i,:,:,:]/176
        
        
    return img.reshape(1,256,256,20)
        
    
    
    
    
a1 = traj(a)
b1 = traj(b)
c1 = traj(c)
d1 = traj(d)
e1 = traj(e)



img = np.zeros([20,256,256]) 
for i in range(0,20):
    
    img[i,:,:] = a[:,:,i]/4 + b[:,:,i]/4 + c[:,:,i]/4 + d[:,:,i]/4 
    
np.save("../data/label_20images1channel.npy",np.transpose(e))

imgs1 = np.concatenate ((train_flair1,train_t11,train_t1c1,train_t21), axis = 3)

plt.imshow(imgs[])    
    
    
    
    
    
    
    
    