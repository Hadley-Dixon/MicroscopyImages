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

X_labeled = npzfile_train['X']
y_labeled = npzfile_train['y']

X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X_labeled, y_labeled, train_size=.8)

# Image display (check)

i = 0
plt.imshow(X_train[i]/255 + .5*y_train[i], cmap='gray')

#%%

class CellDataset():
  def __init__(self, df, mask):
    self.df = df
    self.mask = mask

  def __len__(self):
    return len(self.df)

  def __getitem__(self, i):
    x = self.df[i] / 255.0
    x = x.reshape((1, 128, 128))
    x = torch.tensor(x, dtype=torch.float32)
    
    y = self.df[i]
    y = torch.tensor(y, dtype=torch.float32)

    return x, y

#%%

dataset_train = CellDataset(X_train, y_train)
dataset_val = CellDataset(X_val, y_val)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=False)

#%%

# U-Net Architecture

class UNet(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding='same')
    self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding='same')
    self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding='same')
    self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding='same')
    self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=3, padding='same')
    self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same')
    self.conv7 = torch.nn.Conv2d(256, 512, kernel_size=3, padding='same')
    self.conv8 = torch.nn.Conv2d(512, 512, kernel_size=3, padding='same')
    self.conv9 = torch.nn.Conv2d(512, 1024, kernel_size=3, padding='same')
    self.conv10 = torch.nn.Conv2d(1024, 1024, kernel_size=3, padding='same')
    # self.dense1 = torch.nn.Linear(16*14*14, 10)

    self.relu = torch.nn.ReLU()
    self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):

    # Encode
    
    x = self.conv1(x) # output (64, 128, 128)
    x = self.relu(x)
    x = self.conv2(x) # output (64, 128, 128)
    x = self.relu(x)
    
    copy1 = x
    
    x = self.maxpool(x) # output (64, 64, 64)
    
    x = self.conv3(x) # output (128, 64, 64)
    x = self.relu(x)
    x = self.conv4(x) # output (128, 64, 64)
    x = self.relu(x)
    
    copy2 = x
    
    x = self.maxpool(x) # output (64, 32, 32)
    
    x = self.conv5(x) # output (256, 32, 32)
    x = self.relu(x)
    x = self.conv6(x) # output (256, 32, 32)
    x = self.relu(x)
    
    copy3 = x
    
    x = self.maxpool(x) # output (256, 16, 16)
    
    x = self.conv7(x) # output (512, 16, 16)
    x = self.relu(x)
    x = self.conv8(x) # output (512, 16, 16)
    x = self.relu(x)
    
    copy4 = x
    
    x = self.maxpool(x) # output (256, 8, 8)
    
    # Base
    
    x = self.conv9(x) # output (1024, 8, 168
    x = self.relu(x)
    x = self.conv10(x) # output (1024, 8, 8)
    x = self.relu(x)
    
    # Decode
    
    # TODO
    
    # # Next, a convolutional layer with 8 filters is applied. Then ReLU is applied.
    # x = self.conv2(x) # output (8, 28, 28)
    # x = self.relu(x)

    # # Next, a maxpooling layer is applied. (Use kernel_size=2 and stride=2.)
    # x = self.maxpool(x) # output (8, 14, 14)

    # # Next, a convolutional layer with 16 filters is applied. Then ReLU is applied.
    # x = self.conv3(x) # output (16, 14, 14)
    # x = self.relu(x)

    # # Next, a flattening layer is applied.
    # x = torch.flatten(x, start_dim=1) # output (16*14*14)x1

    # # Finally, a linear (sometimes called "dense") layer is applied.
    # x = self.dense1(x) # output (10x1)

    return