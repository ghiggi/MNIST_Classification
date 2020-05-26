# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:20:49 2020

@author: feldmann
"""
# Import packages
import numpy as np
import skimage.morphology as skim
import skimage.transform as skit
from keras.models import load_model

# Import functions
import feat_ext

#%% Define data path
#path='/home/feldmann/Documents/Classes/Image_analysis_pattern_recognition/group_projects/Image-analysis-and-pattern-recognition/Monika/data/'
path='../data/'
models_folder_path = "../data/models"
file='robot_parcours_1.avi'

#%% Read video, extract first frame
image, video= feat_ext.read_video(path,file)

#%% Filter the arrow
red_blue_ratio = image[:,:,0]/image[:,:,2]
image_red = np.asarray([red_blue_ratio>2])[0,:,:]

y,x,c_y,c_x = feat_ext.biggest_object(image_red)

mask_arrow = feat_ext.mask_centroid(image_red, c_y, c_x, 70)

#%% Filter on red channel to extract digits
image_r_filt = [image[:,:,0]<100]*mask_arrow

#%% Mathematical morphology
kernel = np.ones([1,4,4])
image_mm = skim.binary_dilation(image_r_filt,kernel)

#%% Sort objects by size and filter
bin_labels, labels, n_obj = feat_ext.size_filtered_object(image_mm,80,500)

#%% Extract features from identified objects, determine color
feature_centroids, feature_extracted, feature_color = feat_ext.extract_feature(labels, n_obj, 20, image)

n_features=len(feature_extracted)

feature_extracted = np.asarray(feature_extracted)
feature_extracted_resize = skit.resize(feature_extracted,[n_features,28,28,3])

#%% Final array for image classification
feature_bin=np.asarray([feature_extracted_resize[:,:,:,0]<0.5][0])


##% Load classifier
model_name = 'final_model.h5'
classifier_path = os.path.join(models_folder_path,model_name)
model = load_model(classifier_path)

# %% Predict classes
model.predict_classes(np.expand_dims(obj_dict['binary_patches'], axis=3))   
