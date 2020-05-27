#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:26:26 2020

@author: ghiggi
"""
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD # Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#%% Define CNN 
def CNN1(n_class):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), 
                     activation='relu',
                     kernel_initializer='he_uniform',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, 
                    activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def CNN2(n_class):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model 

def CNN4(n_class):
    cnn4 = Sequential()
    cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    cnn4.add(BatchNormalization())
    
    cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    cnn4.add(BatchNormalization())
    cnn4.add(MaxPooling2D(pool_size=(2, 2)))
    cnn4.add(Dropout(0.25))
    
    cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    cnn4.add(BatchNormalization())
    cnn4.add(Dropout(0.25))
    
    cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    cnn4.add(BatchNormalization())
    cnn4.add(MaxPooling2D(pool_size=(2, 2)))
    cnn4.add(Dropout(0.25))
    
    cnn4.add(Flatten())
    
    cnn4.add(Dense(512, activation='relu'))
    cnn4.add(BatchNormalization())
    cnn4.add(Dropout(0.5))
    
    cnn4.add(Dense(128, activation='relu'))
    cnn4.add(BatchNormalization())
    cnn4.add(Dropout(0.5))
    
    cnn4.add(Dense(n_class, activation='softmax'))
    
    cnn4.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(lr=1e-3),
                 metrics=['accuracy'])
    return(cnn4)

#%% Define callbacks
def get_callbacks(name_weights, patience_lr=10):
    mcp_save = ModelCheckpoint(name_weights, 
                               save_best_only=True,
                               monitor='val_loss',
                               mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', 
                                       factor=0.1,
                                       patience=patience_lr, 
                                       verbose=1, 
                                       epsilon=1e-4,
                                       mode='min')
    return [mcp_save, reduce_lr_loss]

#%% K-fold cross-validation
def cv_evaluation(model, dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # Prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # Enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # Select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # Fit model
        history = model.fit(trainX, trainY, 
                            epochs=10,
                            batch_size=32, 
                            validation_data=(testX, testY),   # validation_split 
                            verbose=0)
        # Evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        # print("%s: %.3f", model.metrics_names[1], acc*100)
        # Store accuracy scores
        scores.append(acc*100)
        histories.append(history)
    
    return scores, histories
  
#%% Plot diagnostic learning curves      
def plot_learning_curve(training, 
                        validation, 
                        ylabel,
                        subplot = None,
                        ax = None, 
                        xlabel='Epoch',
                        title='', 
                        legend=['Training','Validation']):
    # training, validation must be keras history objects 
    if subplot is not None:
        ax = plt.subplot(subplot)
    elif ax is not None:
        ax = ax 
    else:
        raise ValueError("Provide either subplot or ax argument")
    ax.plot(training, color='blue')
    ax.plot(validation, color='orange')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Epoch')
    ax.legend(legend, loc='upper left')
    return None
  
# %% Plot for confusion matrix 
def plot_confusion_matrix(ax,
                          y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          verbose=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    #%% Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    #%% Normalization option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Plotting a normalized confusion matrix")
    else:
        print('Plotting a confusion matrix, without normalization')
    #%% Print confusion matrix 
    if verbose is True:
        print(cm)
    #%% Create the figure 
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    # plt.xlim(-0.5, len(np.unique(y))-0.5)
    # plt.ylim(len(np.unique(y))-0.5, -0.5)
    return ax
