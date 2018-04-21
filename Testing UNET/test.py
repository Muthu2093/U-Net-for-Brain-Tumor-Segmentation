#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 18:41:57 2018

@author: muthuvel
"""
import numpy as np

a = np.array(([1,2,0],
              [-1,2,0],
              [1,-2,0],
              [-1,-2,0],
              [2,1,0],
              [-2,1,0],
              [2,-1,0],
              [2,-1,0]))
for i in range(0,8):
    a[i,2]= a[i,0]*a[i,1]
    a[i,0]= pow(a[i,0],2)
    a[i,1]= pow(a[i,1],2)
    
    
    
    
#for i in range(0,len(a)):
#    for i in range(i+1,len(a)):
        
    16
    34
    34
    