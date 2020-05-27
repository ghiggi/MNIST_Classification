#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:21:03 2020

@author: GHIGGI
"""
#%% FUNCTIONS FOR MAIN AND TRAINING FILES
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import av
import skimage.morphology as skim
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
#%% Reading video, extracting first frame
def read_video(path, file):
    video_container = av.open(os.path.join(path,file))
    video=[]
    for packet in video_container.demux():
        for frame in packet.decode():
            img=frame.to_image()
            video.append(np.asarray(img))   
    image=video[0]
    video = np.asarray(video)
    return image, video

#%% Identifying biggest object in image
def biggest_object(image):
    (labels, n_obj) = ndi.measurements.label(image)
    obj_size=np.zeros(n_obj+1)
    for label in range(1,n_obj+1):
        obj_size[label]=len(np.where(labels==label)[0])
    largest=np.where(obj_size==np.max(obj_size))
    y,x=np.where(labels==largest[0])
    c_x=np.round(np.nanmean(x))
    c_y=np.round(np.nanmean(y))
    return y, x, c_y, c_x

#%% Mask a square around a centroid
def mask_centroid(image, c_y, c_x, size):
    window=int(size)
    x_l=int(c_x-window-1)
    x_u=int(c_x+window)
    y_l=int(c_y-window-1)
    y_u=int(c_y+window)
    mask=np.ones(image.shape)
    mask[y_l:y_u,x_l:x_u]=0
    return mask

#%% Retrieve arrow centroid
def getPositionArrow(image):
    red_blue_ratio = image[:,:,0]/image[:,:,2]
    image_red = np.asarray([red_blue_ratio>2])[0,:,:]
    y,x,c_y,c_x = biggest_object(image_red)    
    return int(c_y),int(c_x)

#%% Retrieve robot tracks
def get_robot_tracks(video):
    tracks = []
    for frame in np.arange(np.size(video,0)):
        tracks.append(getPositionArrow(video[frame,:,:,:])) 
    return(tracks)


#%% Remove arrow
def remove_arrow(image):
    # Filter arrow
    red_blue_ratio = image[:,:,0]/image[:,:,2]
    image_red = np.asarray([red_blue_ratio > 2])[0,:,:]
    # plt.imshow(image_red)
    y,x,c_y,c_x = biggest_object(image_red)
    mask_arrow = mask_centroid(image_red, c_y, c_x, 70)
    #%% Filter on red channel to extract digits
    image_r_filt = [image[:,:,0]<100]*mask_arrow
    # plt.imshow(image_r_filt[0,:,:])
    return(image_r_filt)


#%% Object identification and size filtering -> discard objects below/above certain size
def size_filtered_object(image, low, hi):
    (labels, n_obj) = ndi.measurements.label(image)
    size=np.zeros(n_obj+1)
    for label in range(1,n_obj+1):
        size[label]=len(np.where(labels==label)[0])
    size[size<low]=0
    size[size>hi]=0
    s_l=np.where(size>0)
    s_labels=np.zeros(labels.shape)
    for label in s_l[0]:
        s_labels[labels==label]=1
    bin_labels=s_labels>0  
    (labels, n_obj) = ndi.measurements.label(bin_labels)
    return bin_labels, labels, n_obj

def size_filtered_object1(image, low, hi):
    (labels, n_obj) = ndi.measurements.label(image)
    size=np.zeros(n_obj+1)
    for label in range(1,n_obj+1):
        a=np.where(labels==label)
        if max(a[1])-min(a[1])>100:
            if np.mean(a[2])<50 or np.mean(a[2])>600:
                labels[labels==label]=0
                continue
            else:
                dx=(a[2][-1]-a[2][1])
                dy=(a[1][-1]-a[1][1])
                dxdy=dx/dy
                
                y1=0; n=a[1][1]
                x1=-n*dxdy + a[2][1]
                for y in range(y1,image.shape[1]):
                    x=int(np.round(x1+y*dxdy))
                    labels[0,y,x-5:x+6]=0
                continue
        size[label]=len(np.where(labels==label)[0])
    size[size<low]=0; size[size>hi]=0;
    s_l=np.where(size>0)
    s_labels=np.zeros(labels.shape)
    for label in s_l[0]:
        s_labels[labels==label]=1
    bin_labels=s_labels>0  
    (labels, n_obj) = ndi.measurements.label(bin_labels)
    return bin_labels, labels, n_obj

#%% Object labelling 
def object_labeling(image):
    #%% Mathematical morphology
    kernel=np.ones([1,4,4])
    image_mm=skim.binary_dilation(image,kernel)
    # plt.imshow(image_mm[0,:,:])
    #%% Sort objects by size and filter
    bin_labels, labels, n_obj = size_filtered_object(image_mm,80,500)
    # plt.imshow(bin_labels[0,:,:])
    # plt.imshow(labels[0,:,:])
    labels = np.squeeze(labels)
    return(labels) 
    

#%% Object identification, extract features around centroids of objects, extract properties
from skimage import exposure   # contrast scaling histogram 
from skimage.color import rgb2gray
from skimage.filters import median as median_filter
from skimage.morphology import disk
from skimage.morphology import binary_dilation
from skimage.transform import resize
def process_object_patch(img, patch_size):
    # Convert to gray scale
    img = rgb2gray(img)
    
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    # plt.imshow(img)
    # plt.show()
    
    # Thresholding 
    img = image_thresholding(img, 0.2, invert=True)
    # plt.imshow(img[0,:,:])
    # plt.show()
    
    # Thick the digits by dilation 
    img = binary_dilation(img[0,:,:], disk(1))
    # plt.imshow(img)
    # plt.show()
       
    # Resize: order 0 for binary images 
    im = resize(img, output_shape=(patch_size,patch_size), 
                order=0, 
                anti_aliasing=False)
    # plt.imshow(im)
    # plt.show()
    return(im)

def object_extraction(image, image_labels, patch_size=28):
    # Initialize objects 
    objects_centroids = []
    objects_patches = []
    objects_binary_patches = []
    objects_type = []
    # Get object labels 
    labels = np.unique(image_labels)
    labels = labels[labels != 0 ] # Remove 0 because is the background
    for label in labels:
        # Identify object 
        img_idx = image_labels == label
        # Identify centroid       
        y,x = np.where(img_idx)
        c_x = np.round(np.nanmean(x))
        c_y = np.round(np.nanmean(y))
        objects_centroids.append([c_x,c_y])
        # Define minimum window length for patch extraction 
        max_extent = np.max([np.sum(np.any(img_idx, axis=0)),
                             np.sum(np.any(img_idx, axis=1))])
        # %% Extract a patch around the object 
        window = max_extent
        x_l=int(c_x - window)
        x_u=int(c_x + window)
        y_l=int(c_y - window)
        y_u=int(c_y + window)
        im = image[y_l:y_u, x_l:x_u]     
        #%% Identify object type (digits vs. operator)           
        # - through blue - red ratio
        b_r = np.nanmean((image[y,x,2]/image[y,x,0]).flatten('C'))
        if b_r >= 1.2: # blue
            object_type = 'OPERATOR'
        if b_r < 1.2:  # black
            object_type = 'DIGIT'
        objects_type.append(object_type)
        #%% Rescale the patch
        im_patch = resize(im, output_shape=(patch_size,patch_size), 
                          order=3, 
                          anti_aliasing=True)
        objects_patches.append(im_patch) 
        #%% Process the original patch and get the binary patch rescaled
        im = process_object_patch(im, patch_size=patch_size)
        objects_binary_patches.append(im) 
    #-------------------------------------------------------------------------.
    objects_centroids = np.asarray(objects_centroids)
    objects_patches = np.asarray(objects_patches)
    objects_binary_patches = np.asarray(objects_binary_patches)
    # Create a dictionary   
    obj_dict = {'centroids': objects_centroids,
                'patches': objects_patches,
                'binary_patches': objects_binary_patches,
                'type': objects_type}
    return obj_dict

#%% Image thresholding
def image_thresholding(images, threshold=0.5, invert=False):
    # invert=False --> below threshold set to 1, above threshold set to 0 
    # invert=True --> below threshold set to 0, above threshold set to 1
    if (images.ndim == 2):
        images = np.expand_dims(images, axis=0)
    if (images.ndim == 3):
        images = np.expand_dims(images, axis=3)
    if invert is False:
         binary_images = images[:,:,:,0] >= threshold   
    else:
         binary_images = images[:,:,:,0] <= threshold   
    binary_images = binary_images.astype(np.uint8) 
    return(binary_images)
  
#%% Create minus sign  
def add_minus_sign(obj_dict):
    # - Use equal digits as template to extract minus sign 
    objects_type = obj_dict['type'] 
    binary_patch = obj_dict['binary_patches']   
    object_labels = obj_dict['labels']
    # Ad-hoc code 
    minus_1 = np.zeros(binary_patch[4,:,:].shape)
    minus_1[12:,:] = binary_patch[4,12:,:]
    minus_1 = np.expand_dims(minus_1,axis=0)
    minus_2 = np.zeros(binary_patch[4,:,:].shape)
    minus_2[:12,:] = binary_patch[4,:12,:]
    minus_2 = np.expand_dims(minus_2,axis=0)
    
    minus_3 = np.zeros(binary_patch[3,:,:].shape)
    minus_3[18:,:] = binary_patch[3,18:,:]
    minus_3 = np.expand_dims(minus_3,axis=0)
    
    minus_4 = np.zeros(binary_patch[3,:,:].shape)
    minus_4[:18,:] = binary_patch[3,:18,:]
    minus_4 = np.expand_dims(minus_4,axis=0)
    
    binary_patch = np.concatenate((binary_patch,
                                    minus_1,
                                    minus_2,
                                    minus_3,
                                    minus_4),axis=0)
    # TODO Compute centroids
    # TODO Compute for original image 
    
    # Add stuffs .... 
    objects_type.append('OPERATOR')
    objects_type.append('OPERATOR')
    objects_type.append('OPERATOR')
    objects_type.append('OPERATOR')
    object_labels.append('-')
    object_labels.append('-')
    object_labels.append('-')
    object_labels.append('-')
    # Add to dictionary 
    obj_dict['type'] = objects_type
    obj_dict['binary_patches'] = binary_patch
    obj_dict['labels'] = object_labels
    return(obj_dict)

#%% Data Augmentation of Operators
def DataAugmentation(images, labels, n, subset_labels=None,  
                     rotation=False, include_original=False, plot=False):
    #%% Check image is rank 4 for compatibility with datagen.flow
    if (images.ndim == 3):
        images = np.expand_dims(images, axis=3)
    #%% Subset images to DataAugment
    if (subset_labels is not None):
        idx_operators = [label in subset_labels for label in labels]
        images = images[idx_operators,:,:,:]
        labels = np.asarray(labels)
        labels = labels[np.where(idx_operators)[0]]
    #%% Data generation settings
    if rotation is False:
       rotation_range = 10
    else:
       rotation_range=180
    batch_size=1
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range = rotation_range,
        width_shift_range = 5,  # pixel shifts
        height_shift_range = 3, # pixel shifts
        horizontal_flip=False,
        zoom_range=0.3)
    #%% Create the iterator
    datagen_iter = datagen.flow(x=images,y=np.asarray(labels),batch_size=batch_size)
    #%% Initialize 
    feature_aug=[]
    feature_aug_label=[]
    i = 1 
    for x_batch, y_batch in datagen_iter:
        x_batch.shape
        feature_aug.append(np.squeeze(x_batch))
        feature_aug_label.append(np.squeeze(y_batch))
        #%% Plot generated images
        if (plot is True):
            plt.imshow(x_batch[0,:,:,0], cmap='Greys_r')
            plt.title('Augmentation round '+str(i) +'; label: '+ y_batch[0])
            plt.show()
        #%% Stop data augmentation when generated n images  
        i+=1
        if i > n: 
            break
    #%% Attach generated data to originals
    feature_aug = np.expand_dims(np.asarray(feature_aug), axis=3)
    feature_aug_label = np.asarray(feature_aug_label)
    if (include_original is True):
        images = np.concatenate((images, feature_aug), axis=0)
        labels = np.concatenate((labels, np.asarray(feature_aug_label)), axis=0) 
    else:
        images = feature_aug
        labels = feature_aug_label
    images = image_thresholding(images, 0.3, invert=False)     
    images = np.expand_dims(np.asarray(images), axis=3)     
    return images, labels 

def load_mnist_data(): 
    (trainX, trainY), (testX, testY) = mnist.load_data()
    images = np.concatenate((trainX,testX), axis=0)
    labels = np.concatenate((trainY,testY), axis=0)
    images = images.astype('float32')
    images = images / 255.0
    images = image_thresholding(images, 0.3, invert=False)
    images = np.expand_dims(np.asarray(images), axis=3)
    return images, labels

