#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:45:08 2018

full inv filter

@author: prathmesh
"""


import numpy as np
import cv2

for kernels in range (1,5):
    kernel_filename = 'blur_kernels/Kernel' + str(kernels) + 'G_SingleTile.png'
    h = cv2.imread(kernel_filename,0)
    for images in range (1,5):
        image_filename = 'blurry_images/Blurry' + str(images) + '_' + str(kernels) + '.png'
        img_bgr = cv2.imread(image_filename,1)
        restored = np.zeros(img_bgr.shape)
        
        print(image_filename)
        print(kernel_filename)
        
        #for each channel (R,G,B)
        for i in range (0,3):
            #1.read image and compute fft
            g = img_bgr[:,:,i]
            G =  (np.fft.fft2(g))
            
            #2. pad kernels with zeros and compute fft
            h_padded = np.zeros(g.shape) 
            h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
            H =  (np.fft.fft2(h_padded))
            
            # normalize to [0,1]
            H_norm = H/abs(H.max())
            G_norm = G/abs(G.max())
            F_temp = G_norm/H_norm
            F_norm = F_temp/abs(F_temp.max())
            
            #rescale to original scale
            F_hat  = F_norm*abs(G.max())
            
            # 3. apply Inverse Filter and compute IFFT
#            F_hat = G / H
            f_hat = np.fft.ifft2( F_hat )
            restored[:,:,i] = abs(f_hat)
        
        #write file 
        out_filename = 'image_metrics/restored_' + str(images) + '_' + str(kernels) + '_1' + '.png'
#        out_filename = 'inv_restored/inv_restored' + str(images) + '_' + str(kernels) + '.png'
        cv2.imwrite(out_filename,restored)
