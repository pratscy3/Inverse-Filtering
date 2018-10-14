#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 20:14:19 2018

trunc filter

@author: prathmesh
"""




import numpy as np
import cv2

#radius of the trunc filter, max=400 as images are 800x800
r = 100

for kernels in range (1,2):
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
            G = np.fft.fftshift(np.fft.fft2(g))
            cv2.imwrite('fft_blurry.png',abs(G))
        
            #2. pad kernels with zeros and compute fft
            h = cv2.imread(kernel_filename,0)
            h_padded = np.zeros(g.shape) 
            h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
            H =  np.fft.fftshift(np.fft.fft2(h_padded))
            # normalize to [0,1]
            H_norm = H/abs(H.max())
            G_norm = G/abs(G.max())
            F_temp = G_norm/H_norm
            F_norm = F_temp/abs(F_temp.max())
            
             #rescale to original scale
            F_hat  = F_norm*abs(G.max())
            
            #4. use restored image inside a circle of radius r and degraded image outside it
            circle_img = np.zeros(g.shape)
            cv2.circle(circle_img,(g.shape[0]//2,g.shape[1]//2),r,1,thickness=-1)
            inner_F_hat = circle_img*F_hat
            circle_img = 1 - circle_img
            outer_G = circle_img*G
            
            #5. take ifft
            f_hat = np.fft.ifft2( np.fft.ifftshift(inner_F_hat + outer_G) ) 
            restored[:,:,i] = abs(f_hat)
        out_filename = 'image_metrics/restored_' + str(images) + '_' + str(kernels) + '_2.png'
#        out_filename = 'trunc_restored/tr_restored' + str(images) + '_' + str(kernels) + '_' + str(r) + '.png'
        cv2.imwrite(out_filename,restored)
