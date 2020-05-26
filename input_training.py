#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:20:49 2020

@author: feldmann
"""
import os 

#%% Import packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.utils import to_categorical # One hot encoding 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#%% Set project directories 
proj_path = "/home/ghiggi/Image-analysis-and-pattern-recognition/Gionata/project"
models_folder_path = "/home/ghiggi/Image-analysis-and-pattern-recognition/Common/data/models"
os.chdir(proj_path)
#%% Import functions
import feat_ext
import CNN_models

#%% Define data path
data_folder_path='/home/ghiggi/Image-analysis-and-pattern-recognition/Gionata/data'
# path='../data/'
filename='robot_parcours_1.avi'

#%% Read video, extract first frame
image, video= feat_ext.read_video(data_folder_path,filename)
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
#%% Assign labels to objects 
obj_dict['labels'] = ['2','3','*','=','=','7','7',':','2','3','+']

#%% Perform image thresholding to create binary patches                                                        
obj_dict['binary_patches'] = feat_ext.image_thresholding(obj_dict['patches'])

#%% Define operator dictionary
operator_dict = {'+': 9, 
                 '-': 10, # set to 1 if rotation invariant 
                 '*': 11,
                 ':': 12,
                 '=': 13}

############################# 
## Code for training CNN ####
#############################
#%% Add patches corresponding to minus sign (using the equal sign as template)
obj_dict = feat_ext.add_minus_sign(obj_dict) 

#%% Data augmentation of operators 
images_op, labels_op = feat_ext.DataAugmentation(images=obj_dict['binary_patches'], 
                                                 labels=obj_dict['labels'],
                                                 subset_labels = ['*','=',':','-'],
                                                 n=20000,
                                                 rotation = False, 
                                                 include_original = False,
                                                 plot = False)
labels_op = np.array([operator_dict[label] for label in labels_op])
#%% Data augmentation of digits 
images_digits, labels_digits = feat_ext.DataAugmentation(images=obj_dict['binary_patches'], 
                                                         labels=obj_dict['labels'],
                                                         subset_labels =['2','3','7'],
                                                         n=20000,
                                                         rotation = False, 
                                                         include_original = False,
                                                         plot = False)
#%% Load MNIST digits 
images_mnist, labels_mnist = feat_ext.load_mnist_data()
labels_mnist[labels_mnist == 9] = 6

#%% Combine MNIST and OPERATORS 
images_all = np.concatenate((images_mnist, images_op),axis=0)
labels_all = np.concatenate((labels_mnist, labels_op),axis=0)

# Encoder categories 
# from sklearn.preprocessing import LabelEncoder
# LE_mnist = LabelEncoder()
# LE_mnist.fit(labels_mnist)
# list(LE_mnist.classes_)
 
# Assume integers from 0 to num_classes 
labels_mnist_Y = to_categorical(labels_mnist)
labels_op_Y = to_categorical(labels_op)
labels_all_Y = to_categorical(labels_all)

#%% Create train and test set (random split)
X_train, X_test, Y_train, Y_test = train_test_split(images_all,labels_all_Y,
                                                    test_size = 0.2,
                                                    shuffle = True,
                                                    random_state=42)
#%% Create train and test set (class-stratified split)
X_train, X_test, Y_train, Y_test = train_test_split(images_all,labels_all_Y,
                                                    test_size = 0.2,
                                                    shuffle = True,
                                                    stratify = labels_all,
                                                    random_state=42)                                                
#%% Train MNIST+operator models 
# Load models 
model = CNN_models.CNN4(14)
model.summary()
# Training options 
n_epochs = 1 # 15  
batch_size = 32
validation_split = 0.33
history = model.fit(X_train, Y_train,
                    epochs = n_epochs,
                    batch_size = batch_size, 
                    validation_split = validation_split,
                    verbose=True)

#%% Train MNIST+operator models (rotation invariant)
# model.fit_generator
# - rotated images 
# - batch ... class stratified ... 

#%% Save the model 
model_name = 'final_model.h5'
classifier_path = os.path.join(models_folder_path,model_name)
model.save(classifier_path) 
        
#%% Evaluation on test set 
model.predict_classes(np.expand_dims(obj_dict['binary_patches'], axis=3))   
digit = model.predict_classes(images_mnist[0:2,:,:,:])
digit2 = model.predict_proba(images_mnist[0:2,:,:,:])
y_pred = model.predict_classes(images_all)

#%% Plot diagnostic learning curves 
plt.subplots(figsize=(10,10))
plt.tight_layout()
display_training_curves(history.history['accuracy'],
                        history.history['val_accuracy'], 
                        ylabel='Cross Entropy Loss', 
                        subplot=211)
display_training_curves(history.history['loss'], 
                        history.history['val_loss'], 
                        ylabel='Loss',
                        suplot=212)

fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
display_training_curves(history.history['accuracy'],
                        history.history['val_accuracy'], 
                        ylabel='Classification Accuracy', 
                        ax = axarr[0])
display_training_curves(history.history['loss'], 
                        history.history['val_loss'], 
                        ylabel='Loss',
                        ax=axarr[1])
plt.show()

# %% Plot confusion matrix 
# score = model.evaluate(x_test, y_test, verbose=0)
CNN_models.plot_confusion_matrix(y_true = labels_all, 
                                 y_pred = y_pred, 
                                 classes = np.unique(labels_all),
                                 normalize=False,
                                 title=None,
                                 cmap=plt.cm.Blues)

np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
         
# %% Plot architecture 
from keras.utils import plot_model
simple_architecture = plot_model(model, 
                                 show_shapes=True, 
                                 show_layer_names=False)
simple_architecture.width = 600
simple_architecture










#%% Training with a Data Generator
# datagen = ImageDataGenerator(  
#     rotation_range=10,  
#     zoom_range=0.1, 
#     width_shift_range=0.1, 
#     height_shift_range=0.1
# )
# datagen.fit(X_train)

# # Fit model
# history = model.fit_generator(
#     datagen.flow(X_train, y_train, batch_size=batch_size),
#     validation_data=(X_test, y_test),
#     epochs=epochs, 
#     verbose=2, 
#     workers=12,
#     callbacks=callbacks
# )
    
# model_name
# model_weights 
# model_weights = "final_model_fold" + str(j) + "_weights.h5"
# callbacks = get_callbacks(name_weights = model_weights, patience_lr=10)

# model.fit_generator(generator,
#                     steps_per_epoch=len(X_train_cv)/batch_size,
#                     epochs=15,
#                     shuffle=True,
#                     verbose=1,
#                     validation_data = (X_valid_cv, y_valid_cv),
#                     callbacks = callbacks)

# # %% Callback, checkpoints 
# # Define checkpoints
# checkpoint = ModelCheckpoint(
#     filepath=f'resnet-{int(time.time())}.dhf5',
#     monitor='loss',
#     save_best_only=True
# )
# # Define callbacks
# callbacks = [checkpoint, annealer]

# callbacks = [
#       tf.keras.callbacks.ModelCheckpoint(
#           ckpt_full_path, save_weights_only=True),
#       tf.keras.callbacks.TensorBoard(log_dir=flags_obj.model_dir),
# ]




                                  
 



