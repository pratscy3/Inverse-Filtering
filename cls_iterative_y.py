#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 18:42:34 2018

iterative y cls filter

@author: prathmesh
"""


import numpy as np
import cv2


global G
global H
global P
global F_hat


p = np.array([(0, -1, 0),
              (-1, 4, -1),
              (0, -1, 0)])
p_padded = np.zeros((800,800)) 
p_padded[:p.shape[0],:p.shape[1]] = np.copy(p)
P =  (np.fft.fft2(p_padded))
            
noise_mean = 0
noise_st_dev = 1.01
noise_eu_norm = 800*800*(noise_mean**2 + noise_st_dev**2)
error = 10000

#computes F_hat, r_eu_norm for a given Y,G,H,P
def compute_r_eu_norm(Y):
    global G
    global H_norm
    global P
    global F_hat
        
    
    H2 = (abs(H)**2 + Y*(abs(P)**2))/(np.conjugate(H))
    H2_norm = H2/abs(H2.max())
    
    G_norm = G/abs(G.max())
    F_temp = G_norm / H2_norm
    F_norm = F_temp/abs(F_temp.max())
    F_hat = F_norm*abs(G.max())
    
    R_norm = G_norm - H_norm*F_norm/abs((H_norm*F_norm).max())
    R = R_norm*abs(G.max())
    r = abs(np.fft.ifft2( R ))
    r_eu_norm = (r**2).sum()
#    print(r_eu_norm)
    return r_eu_norm

for kernels in range (1,5):
    kernel_filename = 'blur_kernels/Kernel' + str(kernels) + 'G_SingleTile.png'
    h = cv2.imread(kernel_filename,0)
    h_padded = np.zeros((800,800)) 
    h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
    H =  (np.fft.fft2(h_padded))
    H_norm = H/abs(H.max())
    for images in range (1,5):
        image_filename = 'blurry_images/Blurry' + str(images) + '_' + str(kernels) + '.png'
        img_bgr = cv2.imread(image_filename,1)
        restored = np.zeros(img_bgr.shape)
        
        print('kernel:',kernels,' image:',images)
        Y_rgb = np.zeros(3)
        #for each channel (R,G,B)
        for i in range (0,3):
            #read image and compute FFT
            g = img_bgr[:,:,i]
            G =  (np.fft.fft2(g))
            
            #Iteratively find Y till the constrains are satisfied
            Y =1000000#initial estimate of Y
            r_eu_norm = compute_r_eu_norm(Y)
            Y_old = 0 #this is for ensuring a break if the iteration keeps oscillating between 2 values
            
            while (r_eu_norm > noise_eu_norm + error or r_eu_norm < noise_eu_norm - error):
                if(r_eu_norm > noise_eu_norm + error):
                    #decrease Y
                    Y-=20000
                    print('r = ',r_eu_norm)
                    print('n = ',noise_eu_norm)
                    print('decreasing Y')
                    print('Y = ',Y)
                    print(' ')
                    r_eu_norm = compute_r_eu_norm(Y)
                    #this is for ensuring a break if the iteration keeps oscillating between 2 values
                    Y_local = Y
                    if Y_old != Y_local:
                        Y_old = Y_local
                    else:
                        print('breaking!')
                        break
                elif(r_eu_norm < noise_eu_norm - error):
                    #increase Y
                    Y+=20000
                    print('r = ',r_eu_norm)
                    print('n = ',noise_eu_norm)
                    print('increasing Y')
                    print('Y = ',Y)
                    print(' ')
                    r_eu_norm = compute_r_eu_norm(Y)
                
            #take FFT of the updated F_hat    
            print('constrained satisfied!')
            Y_rgb[i] = Y
            f_hat = np.fft.ifft2( F_hat ) 
            restored[:,:,i] = abs(f_hat)
        Y_av = Y_rgb.mean()
#        out_filename = 'image_metrics/aa_restored_' + str(images) + '_' + str(kernels) + '_4_Y' + str(int(Y_av)) + '.png'
        out_filename = 'cls_restored/cl_restored' + str(images) + '_' + str(kernels) + '_' + str(int(Y)) + '.png'
 
        cv2.imwrite(out_filename,restored)