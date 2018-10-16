#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 12:44:33 2018

wiener filter

@author: prathmesh
"""

import numpy as np
import cv2

#constant estimate
K = 35000*25


for kernels in range (1,5):
    kernel_filename = 'blur_kernels/Kernel' + str(kernels) + 'G_SingleTile.png'
    h = cv2.imread(kernel_filename,0)
    for images in range (1,5):
        image_filename = 'blurry_images/Blurry' + str(images) + '_' + str(kernels) + '.png'
        img_bgr = cv2.imread(image_filename,1)
        restored = np.zeros(img_bgr.shape)
        #for each channel (R,G,B)
        for i in range (0,3):
            #read image and compute FFT
            g = img_bgr[:,:,i]
            G =  (np.fft.fft2(g))
            
            #2. pad kernels with zeros and compute fft
            h = cv2.imread(kernel_filename,0)
            h_padded = np.zeros(g.shape) 
            h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
            H =  (np.fft.fft2(h_padded))
            # normalize to [0,1]
            
            #3. Find the inverse filter term
            weiner_term = (abs(H)**2 + K)/(abs(H)**2)
            print("max value of abs(H)**2 is ",(abs(H)**2).max())
            H_weiner = H*weiner_term
            # normalize to [0,1]
            H_norm = H_weiner/abs(H_weiner.max())
            
            G_norm = G/abs(G.max())
            F_temp = G_norm/H_norm
            F_norm = F_temp/abs(F_temp.max())
            
            #rescale to original scale
            F_hat  = F_norm*abs(G.max())
            
            f_hat = np.fft.ifft2( F_hat )
            restored[:,:,i] = abs(f_hat)
        
        out_filename = 'image_metrics/restored_' + str(images) + '_' + str(kernels) + '_3' + '.png'
#        out_filename = 'weiner_restored/we_restored' + str(images) + '_' + str(kernels) + '_' + str(K) + '.png'
        cv2.imwrite(out_filename,restored)
