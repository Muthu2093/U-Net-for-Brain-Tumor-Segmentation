#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 20:08:45 2018

@author: muthuvel
"""


from medpy.io import load
import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import SimpleITK


#img = nib.load('/Users/muthuvel/Documents/GitHub/Generative-Adversarial-Networks-for-Brain-Tumor-Segmentation/MICCAI_BraTS17_Data_Training/HGG/Brats17_2013_2_1/Brats17_2013_2_1_flair.nii')
#data = img.get_data()
#imgplot = plt.imshow(data)
#plt.show()
#data_path='/Users/muthuvel/Documents/GitHub/Generative-Adversarial-Networks-for-Brain-Tumor-Segmentation/BRATS_2013_Training/Image_Data/HG/0001/VSD.Brain_3more.XX.XX.OT/'
#example_filename = os.path.join(data_path, 'VSD.Brain_3more.XX.XX.OT.6560.mha')

#inputImage = SimpleITK.ReadImage(example_filename)

image_data, image_header = load('/Users/muthuvel/Documents/GitHub/Generative-Adversarial-Networks-for-Brain-Tumor-Segmentation/test.mha')