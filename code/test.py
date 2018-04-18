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
