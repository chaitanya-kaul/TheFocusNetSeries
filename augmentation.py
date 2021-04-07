#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:49:07 2019

@author: ck807
"""

import Augmentor

input_validation_directory = '/home/ck807/isic_2017_experiments/ISIC-2017_Validation_Data/'
input_validation_mask_directory = '/home/ck807/isic_2017_experiments/ISIC-2017_Validation_Part1_GroundTruth/'
input_training_directory = '/home/ck807/isic_2017_experiments/ISIC-2017_Training_Data/'
input_training_mask_directory = '/home/ck807/isic_2017_experiments/ISIC-2017_Training_Part1_GroundTruth/'
batch_s = 16

p = Augmentor.Pipeline(input_training_directory)
p.ground_truth(input_training_mask_directory)
p.rotate(probability=0.2, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.2)
p.zoom_random(probability=0.1, percentage_area=0.8)
p.flip_top_bottom(probability=0.3)
p.gaussian_distortion(probability=0.05, grid_width=4, grid_height=4, magnitude=3, corner='bell', method='in', mex=0.5, mey=0.5, sdx=0.05, sdy=0.05)
p.random_brightness(probability=0.05, min_factor=0.7, max_factor=1.3)
p.random_color(probability=0.05, min_factor=0.6, max_factor=0.9)
p.random_contrast(probability=0.05, min_factor=0.6, max_factor=0.9)
p.random_distortion(probability=0.2, grid_width=4, grid_height=4, magnitude=2)
p.sample(3000)