#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Project: Counting Cells in Microscopy Images
MATH 373

Created on Sun May 12 13:55:33 2024

@author: hadleydixon
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import sklearn.model_selection
import torch
import matplotlib.pyplot as plt

npzfile_train = np.load('/Users/hadleydixon/Desktop/MicroscopyImages/counting-cells-in-microscopy-images-2024/train_data.npz')
npzfile_test = np.load("/Users/hadleydixon/Desktop/MicroscopyImages/counting-cells-in-microscopy-images-2024/test_images.npz")
df_train, df_val = sk.model_selection.train_test_split(npzfile_train, train_size=.8)

X = npzfile_train['X']
y = npzfile_train['y']

# Image display (check)

i = 0
plt.imshow(X[i]/255 + .5*y[i], cmap='gray')

#%%

