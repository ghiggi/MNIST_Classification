# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:20:49 2020

@author: GHIGGI
"""
#%% Set working directory
import os 
# proj_path = "/home/ghiggi/Image-analysis-and-pattern-recognition/Common/project"
# proj_path = "/home/ghiggi/Image-analysis-and-pattern-recognition/Common/project"
proj_path = "/home/ghiggi/Documents/MNIST_Classification"
os.chdir(proj_path)

#%% Import packages
import numpy as np

import skimage.morphology as skim
import skimage.transform as skit
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
print("Tensorflow version " + tf.__version__)

#%% Import functions
import feat_ext # feature extraction
import CNN_models # CNN training tools 
import pp # post-processing

#%% Set project directories 
data_folder_path = os.path.join(proj_path,'data')
models_folder_path = os.path.join(proj_path,'models') 
path_number_imgs = os.path.join(proj_path,'data/numbers_img') 

# path_number_imgs ='../data/numbers_img/' 
# models_folder_path = '../models'
# data_folder_path ='../data/'

#%% Read video and extract first frame
filename='robot_parcours_1.avi'
image, video = feat_ext.read_video(data_folder_path,filename)
# plt.imshow(image)

#%% Remove arrow
image_r_filt = feat_ext.remove_arrow(image)
# plt.imshow(image_r_filt[0,:,:])

#%% Object labelling   
image_labels = feat_ext.object_labeling(image_r_filt)

#%% Extract image patches from identified objects
obj_dict = feat_ext.object_extraction(image = image,
                                      image_labels = image_labels, 
                                      patch_size = 28)

#%% Perform image thresholding to create binary patches                                                        
obj_dict['binary_patches'] = feat_ext.image_thresholding(obj_dict['patches'])

#%% Load CNN model
model_name = 'final_model.h5'
classifier_path = os.path.join(models_folder_path,model_name)
model = load_model(classifier_path)

#%% Predict classes
labels_dict = {'1': 1,
               '2': 2,
               '3': 3,
               '4': 4,
               '5': 5,
               '6': 6,
               '7': 7,
               '8': 8,
               '9': 6, # set to 6 if the model must be rotation invariant
               '+': 9, 
               '-': 1, # set to 1 if the model must be rotation invariant 
               '*': 10,
               ':': 11,
               '=': 12}
obj_dict['labels'] = np.argmax(model.predict(np.expand_dims(obj_dict['binary_patches'], axis=3)), axis=-1) 

#%% Load objects templates dictionary 
number_imgs = pp.load_object_template(path_number_imgs)
#%% Retrieve robot tracks 
robot_tracks = feat_ext.get_robot_tracks(video)

#%% Create Video 
# Add equation 
# - Get_closest_centroid(tracks[frame], centroid)
# - Check_robot_is_over  
# - Get index centroid
# - Assign labels   object_dict['labels'][0] 
# - Check that there is an alternation between digits and operators 
# - Substitute 1 with 13 if it is a minus (when consecutive operators)

obj_dict['centroids']
obj_dict['centroids'][1] # row array

obj_dict['labels']
obj_dict['labels'][0]

video = pp.add_equation(video, tracks, object_dict, number_imgs)

# Add robot tracks
video = pp.add_tracks(video, 
                      tracks = robot_tracks, 
                      cumulative=True)

# Video writing
out = cv2.VideoWriter('video.mp4',cv2.VideoWriter_fourcc(*'H264'), 2, (720, 480))
for i in video:
    out.write(i)
out.release()



