#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 23:19:56 2018

@author: prathmesh
"""
import time
import math
import numpy as np
import cv2
def iexp(n):
    return complex(math.cos(n), math.sin(n))
def dft2d(f):
    one_D_DFT = np.zeros(f.shape,dtype="complex128")
    M = f.shape[1] #horizontal
    N = f.shape[0] #vertical
    for y in range(N):
        one_D_DFT[y,:] = [sum((f[y,x] * iexp(-2 * math.pi * u * x / M) for x in range(M))) for u in range(M)]
    two_D_DFT = [sum((one_D_DFT[y,:]*iexp(-2 * math.pi * v * y / N) for y in range(N))) for v in range(N)]
    return np.array(two_D_DFT)
#    two_D_DFT = sum(sum((f[:,x] * iexp(-2 * math.pi * u * x / M) for x in range(M)))) for y in range(N)
#            for u in range(M)]

def invdft2d(F):
    one_D_IDFT = np.zeros(F.shape,dtype="complex128")
    M = F.shape[1] #horizontal
    N = F.shape[0] #vertical
#    y=0
    for y in range(N):
        one_D_IDFT[y,:] = [sum((F[y,x] * iexp(2 * math.pi * u * x / M) for x in range(M))) for u in range(M)]
    two_D_IDFT = [sum((one_D_IDFT[y,:]*iexp(2 * math.pi * v * y / N) for y in range(N))) for v in range(N)]
    
    return ((np.array(two_D_IDFT)/(M*N)).real)
    
#f = np.array([[1,2,3,4],[2,2,3,4]])
image_filename = 'my_image/my_blur.png'
img_bgr = cv2.imread(image_filename,1)
sig = img_bgr[:,:,0]
#x = np.arange(10)
start_time = time.time()
F = dft2d(sig)
print('time elapsed 1 =',time.time() - start_time)
start_time = time.time()
#G = np.fft.fft2(sig)
f_new = invdft2d(F)
print('time elapsed 2 =',time.time() - start_time)

#print(F)
print('')

