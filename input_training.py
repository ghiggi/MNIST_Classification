#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:20:49 2020

@author: GHIGGI
"""
#%% Set working directory
import os 
# proj_path = "/home/ghiggi/Image-analysis-and-pattern-recognition/Gionata/project"
proj_path = "/home/ghiggi/Documents/MNIST_Classification"
os.chdir(proj_path)

#%% Import packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical # One hot encoding 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("Tensorflow version " + tf.__version__)

#%% Import functions
import feat_ext
import CNN_models
from CNN_models import plot_learning_curve
from CNN_models import plot_confusion_matrix

#%% Set project directories 
data_folder_path = os.path.join(proj_path,'data')
models_folder_path = os.path.join(proj_path,'models')
# models_folder_path = '../models'
# data_folder_path ='../data/'

#%% Read video and extract first frame
filename='robot_parcours_1.avi'
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
# Visualize the object patches 
# for img in obj_dict['patches']:
#   plt.imshow(img)
#   plt.show()
 
# for img in obj_dict['binary_patches']:
#   plt.imshow(img)
#   plt.show()
#%% Assign labels to objects 
obj_dict['labels'] = ['2','3','*','=','=','7','7',':','2','3','+']

#%% Define labels dictionary
labels_dict = {'1': 1,
               '2': 2,
               '3': 3,
               '4': 4,
               '5': 5,
               '6': 6,
               '7': 7,
               '8': 8,
               '9': 6,  # set to 9 if the model must be rotation invariant
               '+': 9, 
               '*': 10,
               ':': 11,
               '=': 12,
               '-': 1}  # set to 1 if the model must be rotation invariant  } 
operator_dict = {'+': 9, 
                 '*': 10,
                 ':': 11,
                 '=': 12,
                 '-': 1 } # set to 1 if the model must be rotation invariant 
#%% Define label encoding
labels_video = np.array([labels_dict[label] for label in obj_dict['labels']])
############################# 
## Code for training CNN ####
#############################
#%% Add patches corresponding to minus sign (using the equal sign as template)
obj_dict = feat_ext.add_minus_sign(obj_dict) 

#%% Data augmentation of operators 
# nearest
images_op, labels_op = feat_ext.DataAugmentation(images=obj_dict['binary_patches'], 
                                                 labels=obj_dict['labels'],
                                                 subset_labels = list(operator_dict.keys()),
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
images_all = images_all.astype('float32')
labels_all = labels_all.astype('int16')

# Check same appearance
# plt.imshow(images_op[0,:,:,0])
# plt.imshow(images_op[20,:,:,0])
# plt.imshow(obj_dict['binary_patches'][4,:,:])
# plt.imshow(images_digits[0,:,:,0])

# plt.imshow(images_mnist[0,:,:,0])
# plt.imshow(images_all[0,:,:,0])

# plt.hist(images_op[0,:,:,0].flatten())
# plt.hist(obj_dict['binary_patches'][4,:,:].flatten())
# plt.hist(images_all[0,:,:,0].flatten())

#%% Create train and test set (random split)
# X_train, X_test, Y_train_labels, Y_test_labels = train_test_split(images_all,labels_all,
#                                                     test_size = 0.2,
#                                                     shuffle = True,
#                                                     random_state=42)
#%% Create train and test set (class-stratified split)
X_train, X_test, Y_train_labels, Y_test_labels = train_test_split(images_all,labels_all,
                                                                  test_size = 0.2,
                                                                  shuffle = True,
                                                                  stratify = labels_all,
                                                                  random_state=42)   
#%% One Hot Encoding Labels
# - Assume integers from 0 to num_classes 
Y_train_OHE = to_categorical(Y_train_labels)
Y_test_OHE = to_categorical(Y_test_labels)                                          
# %% Define CNN model and training options
# Load models 
model = CNN_models.CNN4(13)
model.summary()
# Training options 
n_epochs = 30 
batch_size = 32
validation_split = 0.33

#%% Train MNIST+operator models 
# Fit the model
# history = model.fit(X_train, Y_train_OHE,
#                     epochs = n_epochs,
#                     batch_size = batch_size, 
#                     validation_split = validation_split,
#                     verbose=1)

#%% Train MNIST+operator models (rotation invariant)
# Define Training and Validation
X_train, X_val, Y_train_labels, Y_val_labels = train_test_split(X_train, Y_train_labels,
                                                                test_size = 0.2,
                                                                shuffle = True,
                                                                stratify = Y_train_labels,
                                                                random_state=42)
# One Hot Encoding Labels
# - Assume integers from 0 to num_classes 
Y_train_OHE = to_categorical(Y_train_labels)
Y_val_OHE = to_categorical(Y_val_labels)  

# Define Data Generator
datagen = ImageDataGenerator(  
    rotation_range=180,  
    zoom_range=0.1, 
    width_shift_range=0.1, 
    height_shift_range=0.1
)
trainGen = datagen.flow(X_train, Y_train_OHE, batch_size=batch_size)
valGen = datagen.flow(X_val, Y_val_OHE, batch_size=batch_size)

# Fit the model 
history = model.fit_generator(
    trainGen,
    validation_data = valGen, # (X_valid, Y_valid_OHE),
    steps_per_epoch = len(X_train)//batch_size, ## number of obs per epoch 
    epochs=n_epochs, 
    verbose=1, 
    # callbacks=callbacks
)

#%% Save the model 
model_name = 'final_model.h5'
classifier_path = os.path.join(models_folder_path,model_name)
model.save(classifier_path) 

#%% Load the model 
model = load_model(classifier_path)
            
#%% Evaluation on test set 
print(np.argmax(model.predict(np.expand_dims(obj_dict['binary_patches'], axis=3)), axis=-1)) 
Y_train_labels_pred = np.argmax(model.predict(X_train), axis=-1)   
Y_test_labels_pred = np.argmax(model.predict(X_test), axis=-1)  

#%% Plot diagnostic learning curves 
fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
plot_learning_curve(history.history['accuracy'],
                    history.history['val_accuracy'], 
                    ylabel='Classification Accuracy', 
                    ax = axarr[0])
plot_learning_curve(history.history['loss'], 
                    history.history['val_loss'], 
                    ylabel='Loss',
                    ax=axarr[1])
plt.show()

# %% Plot confusion matrix 
fig, axarr = plt.subplots(1, 2, figsize=(25, 10))
plot_confusion_matrix(ax = axarr[0],
                      y_true = Y_test_labels, 
                      y_pred = Y_test_labels_pred,
                      classes = np.unique(Y_train_labels),
                      normalize=False,
                      title='Confusion matrix without normalization',
                      cmap=plt.cm.Blues)
np.set_printoptions(precision=2)
plot_confusion_matrix(ax = axarr[1],
                      y_true = Y_test_labels, 
                      y_pred = Y_test_labels_pred,
                      classes = np.unique(Y_train_labels),
                      normalize=True,
                      title='Normalized confusion matrix',
                      cmap=plt.cm.Blues)
fig.tight_layout()
plt.show()

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]
model.compile(...,metrics=METRICS)

## TensorFlow addons
# FBetaScore
# https://github.com/tensorflow/addons/blob/master/tensorflow_addons/metrics/f_scores.py
 
# https://github.com/PhilipMay/mltb#module-keras-for-tfkeras


## learning rate scheduling 
## tensoarboard
## callbacks
## compute other metrics 

## Compute loss function for each single obs 
## Display worst loss & gradient 


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)


# Bias initializer in last dense layer when activation sigmoid or softmax for classification
# binary classification: bias_initializer = np.log([pos/neg]) 
#  http://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines



## Plot automatically history metrics 
def plot_metrics(history):
  metrics =  ['loss', 'auc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()
    
def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')

# Class weight for binary classification
# - Do not use optimizers (like SGD) whose step size is dependent on the magnitude of the gradient
weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

# Oversample the minority class
# - Stratified sampling


    
# %% Plot architecture 
# from tensorflow.keras.utils import plot_model
# simple_architecture = plot_model(model, 
#                                  show_shapes=True, 
#                                  show_layer_names=False)
# simple_architecture.width = 600
# simple_architecture









 

# model_name
# model_weights 
# model_weights = "final_model_fold" + str(j) + "_weights.h5"
# callbacks = get_callbacks(name_weights = model_weights, patience_lr=10)

## train_on_batch
 

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


# opt = SGD(lr=1e-2, momentum=0.9, decay=1e-2 / NUM_EPOCHS)

                                  
 



