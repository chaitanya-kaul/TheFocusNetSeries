#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:59:42 2019

@author: ck807
"""

# load the images and the masks
import glob
import os
from PIL import Image

input_training_directory = '/home/ck807/isic_2017_experiments/ISIC-2017_Training_Data/'
input_training_mask_directory = '/home/ck807/isic_2017_experiments/ISIC-2017_Training_Part1_GroundTruth/'

img_files = []
for files in glob.glob(input_training_directory + '*.jpg'):
    img_files.append(files)

mask_files = []
for files in glob.glob(input_training_mask_directory + '*.png'):
    mask_files.append(files)
    
img_files.sort()
mask_files.sort()


# rename all the mask images, exclude _segmentation part, Augmentor
for old_name in mask_files:
  new_name = old_name[:-17] + '.png'
  os.rename(old_name,new_name)

# rename image files to .png -> same extension -> augmentor
for old_name in img_files:
  new_name = old_name[:-4] + '.png'
  os.rename(old_name,new_name)


# load the images and the masks
img_files = []
for files in glob.glob(input_training_directory + '*.png'):
    img_files.append(files)

mask_files = []
for files in glob.glob(input_training_mask_directory + '*.png'):
    mask_files.append(files)

img_files.sort()
mask_files.sort()


for files in glob.glob(input_training_directory + '*.png'):
    im = Image.open(files)
    f, e = os.path.splitext(files)
    imResize = im.resize((256, 192), Image.ANTIALIAS)
    imResize.save(f + '.png', 'PNG', quality=100)


for files in glob.glob(input_training_mask_directory + '*.png'):
    im = Image.open(files)
    f, e = os.path.splitext(files)
    imResize = im.resize((256, 192), Image.ANTIALIAS)
    imResize.save(f + '.png', 'PNG', quality=100)
    
    
