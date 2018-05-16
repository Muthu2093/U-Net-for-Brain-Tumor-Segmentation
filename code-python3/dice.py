#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 12:55:32 2018

@author: muthuvel
"""

intersection = np.logical_and(imgs, imgs_test_label)
2. * intersection.sum() / (imgs.sum() + imgs_test_label.sum())
