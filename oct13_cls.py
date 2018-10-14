#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 19:00:31 2018

costant y cls filter

@author: prathmesh
"""

import numpy as np
import cv2
Y =2000000#initial estimate of Y
p = np.array([(0, -1, 0),
              (-1, 4, -1),
              (0, -1, 0)])
p_padded = np.zeros((800,800)) 
p_padded[:p.shape[0],:p.shape[1]] = np.copy(p)
P =  (np.fft.fft2(p_padded))


for kernels in range (1,2):
    kernel_filename = 'blur_kernels/Kernel' + str(kernels) + 'G_SingleTile.png'
    kernel_filename = 'my_image/estimate_4.png'
    h = cv2.imread(kernel_filename,0)
    h_padded = np.zeros((800,800)) 
    h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
    H =  (np.fft.fft2(h_padded))
    H_norm = abs(H/H.max())
    for images in range (1,2):
        image_filename = 'blurry_images/Blurry' + str(images) + '_' + str(kernels) + '.png'
        image_filename = 'my_image/my_blur.png'
        img_bgr = cv2.imread(image_filename,1)
        restored = np.zeros(img_bgr.shape)
        
        print('kernel:',kernels,' image:',images)
        #for each channel (R,G,B)
        for i in range (0,3):
            #read image and compute FFT
            g = img_bgr[:,:,i]
            G =  (np.fft.fft2(g))

            H2 = (abs(H)**2 + Y*(abs(P)**2))/(np.conjugate(H))
            H2_norm = H2/abs(H2.max())
            
            G_norm = G/abs(G.max())
            F_temp = G_norm / H2_norm
            F_norm = F_temp/abs(F_temp.max())
            F_hat = F_norm*abs(G.max())
            
            f_hat = np.fft.ifft2( F_hat ) 
            restored[:,:,i] = abs(f_hat)
#        out_filename = 'constant_y_cls_res/' + str(Y) + 'ycls_restored_' + str(images) + '_' + str(kernels) + '_4_Y' + str(Y) + '.png'
        out_filename = 'image_metrics/restored_' + str(images) + '_' + str(kernels) + '_4' + '.png'
        out_filename = 'my_image/restored_cls1_' + str(Y) + '.png'
        cv2.imwrite(out_filename,restored)
            
            