#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 22:17:59 2018

@author: prathmesh
"""

import numpy as np
import cv2






kernel_filename = 'estimate_4.png'
h = cv2.imread(kernel_filename,0)

image_filename = 'my_blur.png'
img_bgr = cv2.imread(image_filename,1)
restored = np.zeros(img_bgr.shape)

print(image_filename)
print(kernel_filename)

r=30
K=4000
Y = 90
p = np.array([(0, -1, 0),
              (-1, 4, -1),
              (0, -1, 0)])

for i in range (0,3):
            g = img_bgr[:,:,i]
            G =  (np.fft.fft2(g))
            
            h_padded = np.zeros(g.shape) 
            h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
            H =  (np.fft.fft2(h_padded))
            
            p_padded = np.zeros(g.shape) 
            p_padded[:p.shape[0],:p.shape[1]] = np.copy(p)
            P =  (np.fft.fft2(p_padded)) 

            H2 = (abs(H)**2 + Y*(abs(P)**2))/(np.conjugate(H))
            H_norm = abs(H2/H2.max())
        
            
            # Inverse Filter 
            F_hat = G / H_norm
            #replace division by zero (NaN) with zeroes
            #a = np.nan_to_num(F_hat)
            f_hat = np.fft.ifft2( F_hat ) #- 50*np.ones(g.shape)
            
            restored[:,:,i] = abs(f_hat)
        
out_filename = 'restored_out_cls.png'
cv2.imwrite(out_filename,restored)