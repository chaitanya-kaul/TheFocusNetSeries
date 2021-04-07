# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019

@author: ck807
"""
from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import models as M
import models_with_f as F
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau,LearningRateScheduler
from keras import callbacks
import pickle

from keras import layers
from keras import models

from keras.optimizers import *

import losses


# ===== normalize over the dataset 
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized
       
    
####################################  Load Data #####################################
tr_data    = np.load('data_train.npy')
te_data    = np.load('data_test.npy')
val_data   = np.load('data_val.npy')

tr_mask    = np.load('mask_train.npy')
te_mask    = np.load('mask_test.npy')
val_mask   = np.load('mask_val.npy')

tr_mask    = np.expand_dims(tr_mask, axis=3)
te_mask    = np.expand_dims(te_mask, axis=3)
val_mask   = np.expand_dims(val_mask, axis=3)

print('ISIC18 Dataset loaded')

tr_data   = dataset_normalized(tr_data)
te_data   = dataset_normalized(te_data)
val_data  = dataset_normalized(val_data)

tr_mask   = tr_mask /255.
te_mask   = te_mask /255.
val_mask  = val_mask /255.

print('dataset Normalized')

# Build model
#model = F.unet1()
#model.summary()

print('Training')
batch_size = 4
nb_epoch = 50

####################################  Network #####################################

img_height = 256
img_width = 256
img_channels = 3

cardinality = 4


def residual_network(x):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.
    
    """
        
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y
    
    def attention_block(y, nb_channels_in, _strides):

        y = add_common_layers(y)
        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        y = add_common_layers(y)
        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)

        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = layers.Activation('sigmoid')(y)

        return y

    def skip_block(y, nb_channels_in, _strides):
        skip = y

        skip = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=_strides, padding='same')(skip)
        skip = add_common_layers(skip)
	
        y = add_common_layers(y)
        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        y = add_common_layers(y)
        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)

        y = layers.Add()([skip, y])
        
        return y


    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // (cardinality)

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        attn1 = []
        attn2 = []
        skip1 = []
        skip2 = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            if j == 0:
                attn1 = attention_block(group, _d, _strides)
            if j == 1:
                skip1 = skip_block(group, _d, _strides)
            if j == 2:
                attn2 = attention_block(group, _d, _strides)
            if j == 3:
                skip2 = skip_block(group, _d, _strides)

        skip1 = layers.Multiply()([skip1, attn1])
        y_1 = layers.Conv2D(_d * 2, kernel_size=(1, 1), strides=(1,1), padding='same')(skip1)
        y_1 = add_common_layers(y_1)

        skip2 = layers.Multiply()([skip2, attn2])
        y_2 = layers.Conv2D(_d * 2, kernel_size=(1, 1), strides=(1,1), padding='same')(skip2)
        y_2 = add_common_layers(y_2)

        groups.append(y_1)
        groups.append(y_2)
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        #y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        #y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y

    
    conv1 = layers.Conv2D(32, kernel_size=(5, 5), strides=(1,1), padding='same')(x)
    conv1 = residual_block(conv1, 32, 48, _project_shortcut=True, _strides=(1,1))
    #conv1_d = residual_block(conv1, 32, 32, _project_shortcut=True, _strides=(2,2))
    conv1_d = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
    
    conv2 = residual_block(conv1_d, 48, 96, _project_shortcut=True, _strides=(1,1))
    conv2 = residual_block(conv2, 96, 96, _project_shortcut=False, _strides=(1,1))
    conv2 = residual_block(conv2, 96, 96, _project_shortcut=False, _strides=(1,1))
    #conv2_d = residual_block(conv2, 64, 64, _project_shortcut=False, _strides=(2,2))
    conv2_d = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    
    conv3 = residual_block(conv2_d, 96, 156, _project_shortcut=True, _strides=(1,1))
    conv3 = residual_block(conv3, 156, 156, _project_shortcut=False, _strides=(1,1))
    conv3 = residual_block(conv3, 156, 156, _project_shortcut=False, _strides=(1,1))
    #conv3_d = residual_block(conv3, 128, 128, _project_shortcut=False, _strides=(2,2))
    conv3_d = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3)
    
    conv4 = residual_block(conv3_d, 156, 256, _project_shortcut=True, _strides=(1,1))
    conv4 = residual_block(conv4, 256, 256, _project_shortcut=False, _strides=(1,1))
    conv4 = residual_block(conv4, 256, 256, _project_shortcut=False, _strides=(1,1))
    drop_conv4 = layers.Dropout(0.5)(conv4)
    #conv4_d = residual_block(drop_conv4, 256, 256, _project_shortcut=False, _strides=(2,2))
    conv4_d = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    
    bottleneck = residual_block(conv4_d, 256, 512, _project_shortcut=True, _strides=(1,1))
    bottleneck = residual_block(bottleneck, 512, 512, _project_shortcut=False, _strides=(1,1))
    drop_bottleneck = layers.Dropout(0.5)(bottleneck)
    
    up1 = layers.UpSampling2D(size = (2,2))(bottleneck)
    up1_c = residual_block(up1, 512, 256, _project_shortcut=True, _strides=(1,1))
    merge1 = layers.Add()([conv4, up1_c])
    conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(merge1)
    out1 = layers.Conv2D(128, 1, padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (8,8))(conv5))
    out1 = add_common_layers(out1)
    out1 = layers.Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(out1)
    
    up2 = layers.UpSampling2D(size = (2,2))(conv5)
    up2_c = residual_block(up2, 256, 156, _project_shortcut=True, _strides=(1,1))
    merge2 = layers.Add()([conv3, up2_c])
    conv6 = layers.Conv2D(156, (3, 3), activation='relu', padding='same')(merge2)
    out2 = layers.Conv2D(64, 1, padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (4,4))(conv6))
    out2 = add_common_layers(out2)
    out2 = layers.Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(out2)
    
    up3 = layers.UpSampling2D(size = (2,2))(conv6)
    up3_c = residual_block(up3, 156, 96, _project_shortcut=True, _strides=(1,1))
    merge3 = layers.Add()([conv2, up3_c])
    conv7 = layers.Conv2D(96, (3, 3), activation='relu', padding='same')(merge3)
    out3 = layers.Conv2D(32, 1, padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(conv7))
    out3 = add_common_layers(out3)
    out3 = layers.Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(out3)
    
    up4 = layers.UpSampling2D(size = (2,2))(conv7)
    up4_c = residual_block(up4, 96, 48, _project_shortcut=True, _strides=(1,1))
    merge4 = layers.Add()([conv1, up4_c])
    conv8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(merge4)
    conv9 = layers.Conv2D(9, (3, 3), activation='relu', padding='same')(conv8)
    conv10 = layers.Conv2D(3, (3, 3), activation='relu', padding='same')(conv9)
    out4 = layers.Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv10)

    out_concat = layers.Concatenate(axis=3)([out1, out2, out3, out4])

    out_concat = layers.Conv2D(32, 1, padding = 'same', kernel_initializer = 'he_normal')(out_concat)
    out_concat = add_common_layers(out_concat)

    out = layers.Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(out_concat)
    
    return out
        
image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
network_output = residual_network(image_tensor)
        
model = models.Model(inputs=[image_tensor], outputs=[network_output])
print(model.summary())
model.compile(optimizer=Adam(lr=0.001), loss=losses.all, metrics=[losses.dsc, losses.jacard_coef, losses.tp, losses.tn, 'accuracy'])


import tensorflow as tf
import keras.backend as K

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


# .... Define your model here ....
print(get_flops(model))

def schedule(epoch):
    if epoch<=15:
        return 1e-4
    elif epoch<=25:
        return 1e-5
    elif epoch<=40:
        return 1e-6
    else:
        return 1e-7

lr_schedule= LearningRateScheduler(schedule)
mcp_save = ModelCheckpoint('weight_isic18_focusnetaplha', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(tr_data,tr_mask,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(val_data, val_mask), callbacks=[mcp_save, lr_schedule] )
  
print('Trained model saved')
def save_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    val_dsc = history.history['val_dsc']
    val_jacard_coef = history.history['val_jacard_coef']
    val_accuracy = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join('focusnetalpha.txt'), 'w') as fp: #
        fp.write('epoch\tloss\tval_loss\tval_dsc\tval_jacard_coef\tval_accuracy\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], val_loss[i], val_dsc[i], val_jacard_coef[i], val_accuracy[i]))
        fp.close()
save_history(history)


