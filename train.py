#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:39:11 2019

@author: ck807
"""
import glob
import matplotlib.pyplot as plt
import cv2
import pandas as pd

import numpy as np
from sklearn.metrics import roc_curve, auc

import models
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

input_validation_directory = '/local/data/data/chaitanya/isic_2017_experiments/ISIC-2017_Validation_Data/'
input_validation_mask_directory = '/local/data/data/chaitanya/isic_2017_experiments/ISIC-2017_Validation_Part1_GroundTruth/'
input_training_directory = '/local/data/data/chaitanya/isic_2017_experiments/ISIC-2017_Training_Data/'
input_training_mask_directory = '/local/data/data/chaitanya/isic_2017_experiments/ISIC-2017_Training_Part1_GroundTruth/'
augmented_data_diectory = '/local/data/data/chaitanya/isic_2017_experiments/ISIC-2017_Training_Data/output/'
x = len('/local/data/data/chaitanya/isic_2017_experiments/ISIC-2017_Training_Data/output/')
modelname = 'focusnet'
history_file_name = modelname + '_history'
batch_s = 8
num_epochs = 100

print('Loading training data (Data Augmentation used)...')
# load the training data
training_img_files = []
training_img_masks = []

for file_name in glob.glob(augmented_data_diectory + '*.png'):
  if(file_name[x]=='_'):
    training_img_masks.append(file_name)
  else:
    training_img_files.append(file_name)

training_img_files.sort()
training_img_masks.sort()

training_images1 = np.zeros((len(training_img_files), 192, 256, 3))
for idx, img_name in enumerate(training_img_files):
    training_images1[idx] = plt.imread(img_name)

training_masks1 = np.zeros((len(training_img_masks), 192, 256, 1))
for idx, mask_name in enumerate(training_img_masks):
    mask = cv2.imread(mask_name)
    ret, thresh_img = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    training_masks1[idx,:,:,0] = thresh_img[:,:,0]

training_img_files = []
training_img_masks = []

for files in glob.glob(input_training_directory + '*.png'):
    training_img_files.append(files)

for files in glob.glob(input_training_mask_directory + '*.png'):
    training_img_masks.append(files)
    
training_img_files.sort()
training_img_masks.sort()

training_images2 = np.zeros((len(training_img_files), 192, 256, 3))
for idx, img_name in enumerate(training_img_files):
    training_images2[idx] = plt.imread(img_name)

training_masks2 = np.zeros((len(training_img_masks), 192, 256, 1))
for idx, mask_name in enumerate(training_img_masks):
    mask = cv2.imread(mask_name)
    ret, thresh_img = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    training_masks2[idx,:,:,0] = thresh_img[:,:,0]

training_images = np.zeros((5000, 192, 256, 3))
training_masks = np.zeros((5000, 192, 256, 1))

training_images[0:3000,:,:,:] = training_images1[:,:,:,:]
training_images[3000:5000,:,:,:] = training_images2[:,:,:,:]

training_masks[0:3000,:,:,:] = training_masks1[:,:,:,:]
training_masks[3000:5000,:,:,:] = training_masks2[:,:,:,:]

training_masks /= 255.

print('Loading validation data...')
# load the validation images and the masks
img_files = []
for files in glob.glob(input_validation_directory + '*.png'):
    img_files.append(files)

mask_files = []
for files in glob.glob(input_validation_mask_directory + '*.png'):
    mask_files.append(files)

img_files.sort()
mask_files.sort()

validation_images = np.zeros((len(img_files), 192, 256, 3))
for idx, img_name in enumerate(img_files):
    validation_images[idx] = plt.imread(img_name)

validation_masks = np.zeros((len(mask_files), 192, 256, 1))
for idx, mask_name in enumerate(mask_files):
    mask = cv2.imread(mask_name)
    ret, thresh_img = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    validation_masks[idx,:,:,0] = thresh_img[:,:,0]

validation_masks /= 255.

print('training model..')
#model = models.unet(input_size = (192,256,3))
#model = models.unet1()
#model = models.attn_unet()
#model = models.wU_Net(192,256,3)
#model = models.Nest_Net(192,256,3)
model = models.focusnet()

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, mode='auto', epsilon=0.1, cooldown=0, min_lr=0.5e-7),
    #ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
    ModelCheckpoint(modelname + '_best_weight.h5', monitor='val_loss', save_best_only=True, verbose=1),
]

h = model.fit(training_images, training_masks, validation_data=(validation_images, validation_masks), 
              shuffle=True, epochs=num_epochs, batch_size=batch_s, 
              verbose=True, callbacks=callbacks)

# save model history to csv
hist_csv_file = history_file_name + '.csv'
hist_df = pd.DataFrame(h.history)
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# plots for model
plt.plot(h.history['dsc'])
plt.plot(h.history['val_dsc'])
plt.title('model Dice')
plt.ylabel('Dice')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'val'], loc='best')
plt.show()
plt.savefig(modelname + '_dsc.png')

plt.plot(h.history['jacard_coef'])
plt.plot(h.history['val_jacard_coef'])
plt.title('model Jaccard')
plt.ylabel('Jaccard')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'val'], loc='best')
plt.show()
plt.savefig(modelname + '_jac.png')

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'val'], loc='best')
plt.show()
plt.savefig(modelname + '_lr.png')

# roc plot
y_pred = model.predict(validation_images)
y_pred = y_pred.ravel()
y_true = validation_masks.ravel()
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
auc = auc(fpr, tpr)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label= 'U-Net (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.grid()
plt.legend(loc='best')
plt.show()
plt.savefig(modelname + '_auc.png')

plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='U-Net (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.grid()
plt.legend(loc='best')
plt.show()
plt.savefig(modelname + '_auc_zoomed.png')
