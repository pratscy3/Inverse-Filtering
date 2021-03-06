#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 18:37:43 2018
WORKS
@author: prathmesh
"""

import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import cv2

#made global so that all functions can modify their value
global canvas
global img_bgr
global restored
global kernel_filename
global text

#flag to detect if file is loaded
global file_loaded
file_loaded=0

#color to be given to buttons
colorval = "#%02x%02x%02x" % (153, 255, 204)

def load_img():
    
    global canvas
    global file_loaded
    global img_bgr
    global restored
    global window
    global kernel_filename
    global text
    
    #presents a dialog box to select file
    #filename (with full path) is extracted from it
    filename = (filedialog.askopenfile(mode="r", initialdir="/home/prathmesh/Desktop/assg2/blurry_images", title="Select Image",\
    filetypes=(("Image files", "*.png"), ("all files", "*.*")))).name

    img_bgr = cv2.imread(filename,1)                         #opencv read, 1: colour 0:grayscale
    print('loaded image')
    restored = np.copy(img_bgr)
    displayl(img_bgr) #display on canvas
    displayr(restored)
    file_loaded=1                                           #make this flag 1
#    All functions first check this global flag
    kernel_filename = (filedialog.askopenfile(mode="r", title="Select Kernel", initialdir="/home/prathmesh/Desktop/assg2/blur_kernels",\
    filetypes=(("Image files", "*.png"), ("all files", "*.*")))).name
                                       
    text.insert(tk.INSERT, "Image and Kernel loaded")
        

        
#converts a cv image (HSV) to tk image and displays it on left side of canvas
def displayl(bgr):
    global img
    global canvas
    canvas.delete("all")
    small = cv2.resize(bgr, (0,0), fx=0.6, fy=0.6)
    img_inter1 = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)   #BGR-->RGB
    img_inter2 = Image.fromarray(img_inter1)            #RGB-->PIL
    img = ImageTk.PhotoImage(img_inter2)                #PIL-->ImageTk
    canvas.create_image(275, 275, image=img)
    print('original img loaded')

#converts a cv image (HSV) to tk image and displays it on right side of canvas
def displayr(bgr):
    global img2
    global canvas2
    canvas2.delete("all")
    small = cv2.resize(bgr, (0,0), fx=0.6, fy=0.6)
    img_inter1 = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)   #BGR-->RGB
    img_inter2 = Image.fromarray(img_inter1)            #RGB-->PIL
    img2 = ImageTk.PhotoImage(img_inter2)                #PIL-->ImageTk
    canvas2.create_image(275, 275, image=img2)
    print('filtered img loaded')
        
def save_img():
    
    global restored
    global file_loaded
    if file_loaded==1:
    
        #presents a dialog box to save file
        #filename (with full path) is extracted from it
        filename = (filedialog.asksaveasfile(mode='w',\
        filetypes=(("Image files", "*.png"), ("all files", "*.*")))).name
        cv2.imwrite(filename,restored)                        #saves file
        print("saved it")

def inv_filter():
    
    global kernel_filename 
    for i in range (0,3):
        global img_bgr
        global restored
        g = img_bgr[:,:,i]
        G =  (np.fft.fft2(g))
        
        h = cv2.imread(kernel_filename,0)
        h_padded = np.zeros(g.shape) 
        h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
        H = (np.fft.fft2(h_padded))
        
        # normalize to [0,1]
        H_norm = H/abs(H.max())
        G_norm = G/abs(G.max())
        F_temp = G_norm/H_norm
        F_norm = F_temp/abs(F_temp.max())
        
        #rescale to original scale
        F_hat  = F_norm*abs(G.max())
        
        # 3. apply Inverse Filter and compute IFFT
        f_hat = np.fft.ifft2( F_hat )
        restored[:,:,i] = abs(f_hat)
    displayr(restored) #display on canvas
 
#creates a new window with entry and button to input blur value
def create_window_for_trunc():
    global window
    global file_loaded
    if file_loaded==1:
        
        window = tk.Toplevel(r)
        window.title(' ')
        window.geometry("700x160")
        window.resizable(width=False,height=False) #not resizable
        entry_trunc=tk.Entry(window)
        entry_trunc.grid(row=2, column=1, columnspan=1,ipady=10)
        bt = tk.Button(window, text='Enter', width=15, command=lambda:trunc_filter(int(entry_trunc.get())))
        bt.grid(row=2, column=2, columnspan=1,ipady=10)
        bt.configure(background=colorval)
        text=tk.Text(window,height=3,width=100)
        text.insert(tk.INSERT, "Enter a radius value between 0 and 400")
        
        text.grid(row=1, column=1, columnspan=8, pady=15,ipady=10)    
        
def trunc_filter(radius):
    global img_bgr
    global restored
#    print(radius)
    for i in range (0,3):
        g = img_bgr[:,:,i]
        G = np.fft.fftshift(np.fft.fft2(g))
    
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
        cv2.circle(circle_img,(g.shape[0]//2,g.shape[1]//2),radius,1,thickness=-1)
        inner_F_hat = circle_img*F_hat
        circle_img = 1 - circle_img
        outer_G = circle_img*G

        f_hat = np.fft.ifft2( np.fft.ifftshift(inner_F_hat + outer_G) ) 
        restored[:,:,i] = abs(f_hat)

    displayr(restored) #display on canvas
    window.destroy()

#creates a new window with entry and button to input blur value
def create_window_for_weiner():
    global window
    global file_loaded
    if file_loaded==1:
        
        window = tk.Toplevel(r)
        window.title(' ')
        window.geometry("700x160")
        window.resizable(width=False,height=False) #not resizable
        entry_weiner=tk.Entry(window)
        entry_weiner.grid(row=2, column=1, columnspan=1,ipady=10)
        bt = tk.Button(window, text='Enter', width=15, command=lambda:weiner_filter(int(entry_weiner.get())))
        bt.grid(row=2, column=2, columnspan=1,ipady=10)
        bt.configure(background=colorval)
        text=tk.Text(window,height=3,width=100)
        text.insert(tk.INSERT, "K = 25 gives good results")
        
        text.grid(row=1, column=1, columnspan=8, pady=15,ipady=10)    

def weiner_filter(K_small):
    global img_bgr
    global restored
    K = K_small*35000
    for i in range (0,3):
        g = img_bgr[:,:,i]
        G =  (np.fft.fft2(g))
        
        h = cv2.imread(kernel_filename,0)
        h_padded = np.zeros(g.shape) 
        h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
        H =  (np.fft.fft2(h_padded))
        
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
    displayr(restored) #display on canvas
    window.destroy()
    
def create_window_for_cls():
    global window
    global file_loaded
    if file_loaded==1:
        
        window = tk.Toplevel(r)
        window.title(' ')
        window.geometry("700x160")
        window.resizable(width=False,height=False) #not resizable
        entry_cls=tk.Entry(window)
        entry_cls.grid(row=2, column=1, columnspan=1,ipady=10)
        bt = tk.Button(window, text='Enter', width=15, command=lambda:cls_filter_constant_Y(int(entry_cls.get())))
        bt.grid(row=2, column=2, columnspan=1,ipady=10)
        bt.configure(background=colorval)
        text=tk.Text(window,height=3,width=100)
        text.insert(tk.INSERT, "Enter an intial Y value. The program will iteratively find \n the optimum Y value. \n Y = 2e6 for K1, 1e6 for others")
        
        text.grid(row=1, column=1, columnspan=8, pady=15,ipady=10)


def cls_filter_constant_Y(Y):
    global img_bgr
    global restored
    for i in range (0,3):
        g = img_bgr[:,:,i]
        G =  (np.fft.fft2(g))
        h = cv2.imread(kernel_filename,0)
        h_padded = np.zeros((800,800)) 
        h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
        H =  (np.fft.fft2(h_padded))
        
        p = np.array([(0, -1, 0),
                  (-1, 4, -1),
                  (0, -1, 0)])
        p_padded = np.zeros((800,800)) 
        p_padded[:p.shape[0],:p.shape[1]] = np.copy(p)
        P =  (np.fft.fft2(p_padded))
        
        H2 = (abs(H)**2 + Y*(abs(P)**2))/(np.conjugate(H))
        H2_norm = H2/abs(H2.max())
        
        G_norm = G/abs(G.max())
        F_temp = G_norm / H2_norm
        F_norm = F_temp/abs(F_temp.max())
        F_hat = F_norm*abs(G.max())
        
        f_hat = np.fft.ifft2( F_hat ) 
        restored[:,:,i] = abs(f_hat)
    displayr(restored) #display on canvas
    window.destroy()
    
    
def compute_r_eu_norm(G,H,P,Y):
    
    H2 = (abs(H)**2 + Y*(abs(P)**2))/(np.conjugate(H))
    H2_norm = H2/abs(H2.max())
    
    G_norm = G/abs(G.max())
    F_temp = G_norm / H2_norm
    F_norm = F_temp/abs(F_temp.max())
    F_hat = F_norm*abs(G.max())
    
    H_norm = H/abs(H.max())
    R_norm = G_norm - H_norm*F_norm/abs((H_norm*F_norm).max())
    R = R_norm*abs(G.max())
    r = abs(np.fft.ifft2( R ))
    r_eu_norm = (r**2).sum()

    return r_eu_norm, F_hat


def cls_filter(Y_in):
    global img_bgr
    global restored
    h = cv2.imread(kernel_filename,0)
    h_padded = np.zeros((800,800)) 
    h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
    H =  (np.fft.fft2(h_padded))
    
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
    
    for i in range (0,3):
        g = img_bgr[:,:,i]
        G =  (np.fft.fft2(g))
        Y = Y_in
        r_eu_norm,F_hat = compute_r_eu_norm(G,H,P,Y)
        Y_old = 0
        while (r_eu_norm > noise_eu_norm + error or r_eu_norm < noise_eu_norm - error):
            if(r_eu_norm > noise_eu_norm + error):
                #decrease Y
                Y-=20000
                print('r = ',r_eu_norm)
                print('n = ',noise_eu_norm)
                print('decreasing Y')
                print('Y = ',Y)
                print(' ')
                r_eu_norm,F_hat = compute_r_eu_norm(G,H,P,Y)
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
                r_eu_norm,F_hat = compute_r_eu_norm(G,H,P,Y)
            
        print('constrained satisfied!')
        f_hat = np.fft.ifft2( F_hat ) 
        
        restored[:,:,i] = abs(f_hat)
    displayr(restored) #display on canvas
    window.destroy()
    
print("start")

#Root window attributes
r = tk.Tk()
r.title('EE610 Image Editor')
r.geometry("1100x700")
r.resizable(width=False,height=False) #not resizable
r.option_add("*Font", "helvetica 11 bold italic") #font for all widgets
main_colorval = "#%02x%02x%02x" % (60,60,60) #background colour for canvas: dark gray
r.configure(background=main_colorval)



#Buttons above canvas
bt_load = tk.Button(r, text='Load Image', width=15, command=load_img)
bt_load.grid(row=1, column=1, columnspan=1, padx=10, pady=15, ipady=5)
bt_load.configure(background=colorval)


bt_save = tk.Button(r, text='Save Image', width=15, command=save_img)
bt_save.grid(row=1, column=4, columnspan=1, padx=10, ipady=5)
bt_save.configure(background=colorval)

text=tk.Text(r,height=1,width=50)
text.grid(row=1, column=2, columnspan=2, pady=15,ipady=10)

#Buttons below canvas
bt_full = tk.Button(r, text='Full Inverse', width=15, command=inv_filter)
bt_full.grid(row=3, column=1, columnspan=1,padx=20, ipady=5, pady=15)
bt_full.configure(background=colorval)

bt_trunc = tk.Button(r, text='Truncated Inverse', width=15, command=create_window_for_trunc)
bt_trunc.grid(row=3, column=2, columnspan=1,padx=20, ipady=5)
bt_trunc.configure(background=colorval)

bt_wiener = tk.Button(r, text='Wiener', width=15, command=create_window_for_weiner)
bt_wiener.grid(row=3, column=3, columnspan=1,padx=20, ipady=5)
bt_wiener.configure(background=colorval)

bt_cls = tk.Button(r, text='constrained least square', width=15, command=create_window_for_cls)
bt_cls.grid(row=3, column=4, columnspan=1,padx=20, ipady=5)
bt_cls.configure(background=colorval)



#Canvas
canvas = tk.Canvas(r,width=550,height=550,bg=colorval)
canvas.grid(row=2, column=1, columnspan=2 ,sticky='w')
canvas2 = tk.Canvas(r,width=550,height=550,bg=colorval)
canvas2.grid(row=2, column=3, columnspan=2 ,sticky='e')

r.mainloop()
    

