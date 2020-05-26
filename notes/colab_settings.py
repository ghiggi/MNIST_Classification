#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:09:01 2020

@author: ghiggi
"""

# %tensorflow_version 2.x
import tensorflow as tf
print("Tensorflow version " + tf.__version__)

device_name = tf.test.gpu_device_name()

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

 
#%% Clear session 
tf.keras.backend.clear_session()
  
#%% Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    #tpu = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    # raise BaseException('ERROR: Not connected to a TPU runtime')
    tpu = None
    gpus = tf.config.experimental.list_logical_devices("GPU")

#%% Set strategy
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    # tf.contrib.distribute.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    # strategy = tf.contrib.distribute.TPUStrategy(tpu)
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  
elif len(gpus) > 1: # multiple GPUs in one VM
    # strategy = tf.distribute.MirroredStrategy(gpus)
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
  strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
  print('Running on single GPU ', gpus[0].name)
else:
  strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
  print('Running on CPU')
print("Number of accelerators: ", strategy.num_replicas_in_sync)

#%% TPU mixed precision
# On TPU, bfloat16/float32 mixed precision is automatically used in TPU computations.
# Enabling it in Keras also stores relevant variables in bfloat16 format (memory optimization).
# On GPU, specifically V100, mixed precision must be enabled for hardware TensorCores to be used.
# XLA compilation must be enabled for this to work. (On TPU, XLA compilation is the default)
MIXED_PRECISION = False
if MIXED_PRECISION:
    if tpu: 
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    else: #
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.config.optimizer.set_jit(True) # XLA compilation
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print('Mixed precision enabled')

#%% Batch and learning rate settings
if strategy.num_replicas_in_sync == 8: # TPU or 8xGPU
    BATCH_SIZE = 16 * strategy.num_replicas_in_sync
    VALIDATION_BATCH_SIZE = 16 * strategy.num_replicas_in_sync
    start_lr = 0.00001
    min_lr = 0.00001
    max_lr = 0.00005 * strategy.num_replicas_in_sync
    rampup_epochs = 5
    sustain_epochs = 0
    exp_decay = .8
elif strategy.num_replicas_in_sync == 1: # single GPU
    BATCH_SIZE = 16
    VALIDATION_BATCH_SIZE = 16
    start_lr = 0.00001
    min_lr = 0.00001
    max_lr = 0.0002
    rampup_epochs = 5
    sustain_epochs = 0
    exp_decay = .8
else: # TPU pod
    BATCH_SIZE = 8 * strategy.num_replicas_in_sync
    VALIDATION_BATCH_SIZE = 8 * strategy.num_replicas_in_sync
    start_lr = 0.00001
    min_lr = 0.00001
    max_lr = 0.00002 * strategy.num_replicas_in_sync
    rampup_epochs = 7
    sustain_epochs = 0
    exp_decay = .8

#%% To avoid bottlenecking on data transfer, stored input data in TFRecord files
# --> 230 images per file


#%% Optimize input loading
AUTO = tf.data.experimental.AUTOTUNE
batch_size = 16 * tpu_strategy.num_replicas_in_sync

#%% Create a distribution strategy
## TPUstrategy scope 
# - Creating the model in the TPUStrategy scope means we will train the model on the TPU
with tpu_strategy.scope(): 
  # Load or define model within the with scope 
  model = create_model()
 

max_lr = 0.00005 * tpu_strategy.num_replicas_in_sync

# with tf.device('/cpu:0'):
# with tf.device('/device:GPU:0'):
    
