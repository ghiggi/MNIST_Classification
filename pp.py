#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:08:18 2020

@author: ghiggi
"""

# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
import pandas as pd
import skimage.io
from skimage.transform import resize
import os

#%% Images of the same size 
def resize_image_list(l_images):
    # Retrieve max width and height
    width_max = 0
    height_max = 0
    for img in l_images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)
    # List with images of the same size 
    images_resized = []
    for img in l_images:

        img = resize(img, output_shape=(height_max, width_max))
        images_resized.append(img)
    return(images_resized)

def load_object_template(path_number_imgs):
    number_imgs = {}
    number_filenames = os.listdir(path_number_imgs)
    number_classes = [int(s.split('.', 1)[0]) for s in number_filenames]
    # Load templates
    for (i,filename) in enumerate(number_filenames):       
        number_imgs[i] = skimage.io.imread(os.path.join(path_number_imgs,filename))    
    # Make templates of the same size
    number_imgs = resize_image_list(number_imgs)    
    number_imgs = zip(dict(number_classes, number_imgs))
    return(number_imgs)

#%% Convert labels to corresponding operator
def format_labels(i):
    if i == '9':
        i = '+'
    elif i == '10':
        i = '/'
    elif i == '11':
        i = '*'
    elif i == '12':
        i = '='
    elif i == '13':
        i = '-'     
    return i
    
# Compute the mathematical equation 
def compute_expression(labels):    
    equation = list(map(format_labels, labels))
    eqS = ''.join(map(str, equation))
    result = str(int(eval(eqS[:-1])))
    resultList = list(result)
    return resultList

# detection [x x x x 1 x x x x - x x x 3 x x x x + ]

def add_equation(video, detection, number_imgs):
    # Define position for adding the the number
    posy = 400  
    posxI = 50 
    # Number and figures
    labels = [] 
    # Get size of template 
    nimg_size = np.array([np.size(number_imgs[0],1), np.size(number_imgs[0],2)])
    # Loop over the video frames 
    for i in np.arange(np.size(video,0)):
        # Retrieve object of current frame
        current_label = detection[i]
        # Set position where add digits 
        posx = posxI #
        # Check if it's however a new object 
        if (current_label != 'x'):
            labels.append(current_label)
            # If the equal sign is detected, compute the equation results 
            if(current_label == '12'):
                result = compute_expression(labels)
                # Add the results 
                labels += result               
        # Add the temporary equation to the video         
        for j in labels:
            video[i, posy:posy+nimg_size[0], posx:posx+nimg_size[1],:] = number_imgs[str(j)]
            posx += nimg_size[1] + 10
    return(video)

#%% Add robot tracks
def add_tracks(video, tracks, cumulative=False): 
    squareR = np.empty((20,20,3))
    squareR[:,:,0] = 50
    squareR[:,:,1] = 205
    squareR[:,:,2] = 50
    if cumulative is True: 
        for frame in np.arange(np.size(video,0)):  
            for (Ry,Rx) in tracks[0:(frame+1)]:        
                video[frame, Ry-10:Ry+10, Rx-10:Rx+10, :] = squareR  
    else:
        for frame in np.arange(np.size(video,0)):  
            (Ry,Rx) = tracks[frame]      
            video[frame, Ry-10:Ry+10, Rx-10:Rx+10, :] = squareR  
    return(video)  

 
 


     


## TODO:
# feat_ext.pad_images_to_same_size
# - replace opencv2 with skimage ??? I have troubles with opencv on my laptop
# - not better to resize to the largest observed extent ? 
# - with option argument : squared=True
#%% Pad images to have same size 
# def pad_images_to_same_size(images):
#     """
#     :param images: sequence of images
#     :return: list of images padded so that all images have same width and height (max width and height are used)
#     """
#     width_max = 0
#     height_max = 0
#     for img in images:
#         h, w = img.shape[:2]
#         width_max = max(width_max, w)
#         height_max = max(height_max, h)

#     images_padded = []
#     for img in images:
#         h, w = img.shape[:2]
#         diff_vert = height_max - h
#         pad_top = diff_vert//2
#         pad_bottom = diff_vert - pad_top
#         diff_hori = width_max - w
#         pad_left = diff_hori//2
#         pad_right = diff_hori - pad_left
#         img_padded = cv2.copyMakeBorder(img,
#                                         pad_top, pad_bottom, pad_left, pad_right,
#                                         cv2.BORDER_CONSTANT, value=0)
#         assert img_padded.shape[:2] == (height_max, width_max)
#         images_padded.append(img_padded)
#     return images_padded
    
