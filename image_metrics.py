#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:17:37 2018

@author: prathmesh
"""

import numpy as np
import cv2
from skimage.measure import compare_ssim


for images in range (1,5):
    gt_filename = 'ground_truths/GroundTruth' + str(images) + '_1_1.jpg'
    gt = cv2.imread(gt_filename,1)
#    for latex table creation
    k=0
    if k==0:
        print('\hline')
        print(images, end =" ")
        k+=1
    for kernels in range (1,5):
#        for latex table creation
        print(' & ',kernels, end =" ")
        i=0
        for technique in range (1,5):
            testimage_filename = 'image_metrics/restored_' + str(images) + '_' + str(kernels) + '_' + str(technique) + '.png'
            testimage = cv2.imread(testimage_filename,1)
            SE = (gt - testimage)**2
            MSE = SE.mean()
            PSNR = 10*np.log10(255*255/MSE)
#            print('Image:',images,'Kernel:',kernels,'Technique:',technique,'PSNR=',PSNR)
            gray_testimage = cv2.cvtColor(testimage, cv2.COLOR_BGR2GRAY)
            gray_gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
            (score, diff) = compare_ssim(gray_testimage, gray_gt, full=True)
#            print('SSIM score:',score) #score can be [-1:1] 1 being perfect match
#            print('')
            
#            for latex table creation
            print(' & ',round(PSNR,3),end=" ")
            i+=1
            if i==4:
                print('\\\ ')