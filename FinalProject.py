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
from google.colab import drive
import os
import cv2

drive.mount('/content/gdrive')
os.chdir('/content/gdrive/MyDrive/Cells (1)')

df_labeled = np.load('Copy of train_data.npz')
df_test = np.load('Copy of test_images.npz')
X_labeled = df_labeled['X']
y_labeled = df_labeled['y']
X_test = df_test['X']

X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X_labeled, y_labeled, train_size=.8)

# Image display (check)

plt.imshow(X_train[0], cmap = 'gray')
plt.title('X Image')
plt.show()

plt.imshow(y_train[0], cmap = 'gray')
plt.title('y Image')
plt.show()

#%%

class CellDataset():
  def __init__(self, df, mask):
    self.df = df
    self.mask = mask

  def __len__(self):
    return len(self.df)

  def __getitem__(self, i):
    x = self.df[i]
    y = self.mask[i]

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0) / 255.0
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

    return x, y

class CellDataset_Test():
  def __init__(self, df):
    self.df = df

  def __len__(self):
    return len(self.df)

  def __getitem__(self, i):
    x = self.df[i]
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0) / 255.0

    return x

#%%

dataset_train = CellDataset(X_train, y_train)
dataset_val = CellDataset(X_val, y_val)
dataset_test = CellDataset_Test(X_test)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle = True)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle = False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle = False)

# Dataloader (check)

x_batch, y_batch = next(iter(dataloader_val))
x_batch.shape

#%%

# U-Net Architecture --> move from collab

class UNet(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.relu = torch.nn.ReLU()
    self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    # Encode

    self.conv1 = torch.nn.Conv2d(1, 64, kernel_size = 3, padding ='same')
    self.conv2 = torch.nn.Conv2d(64, 64, kernel_size = 3, padding ='same')
    self.conv3 = torch.nn.Conv2d(64, 128, kernel_size = 3, padding ='same')
    self.conv4 = torch.nn.Conv2d(128, 128, kernel_size = 3, padding ='same')
    self.conv5 = torch.nn.Conv2d(128, 256, kernel_size = 3, padding ='same')
    self.conv6 = torch.nn.Conv2d(256, 256, kernel_size = 3, padding ='same')
    self.conv7 = torch.nn.Conv2d(256, 512, kernel_size = 3, padding ='same')
    self.conv8 = torch.nn.Conv2d(512, 512, kernel_size = 3, padding ='same')
    self.conv9 = torch.nn.Conv2d(512, 1024, kernel_size = 3, padding='same')
    self.conv10 = torch.nn.Conv2d(1024, 1024, kernel_size = 3, padding='same')

    # Decode

    self.upConv1 = torch.nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride=2, padding = 0)
    self.upConv2 = torch.nn.ConvTranspose2d(512, 256, kernel_size = 2, stride=2, padding = 0)
    self.upConv3 = torch.nn.ConvTranspose2d(256, 128, kernel_size = 2, stride=2, padding = 0)
    self.upConv4 = torch.nn.ConvTranspose2d(128, 64, kernel_size = 2, stride=2, padding = 0)

    self.conv11 = torch.nn.Conv2d(1024, 512, kernel_size = 3, padding = 'same')
    self.conv12 = torch.nn.Conv2d(512, 512, kernel_size = 3, padding = 'same')
    self.conv13 = torch.nn.Conv2d(512, 256, kernel_size = 3, padding = 'same')
    self.conv14 = torch.nn.Conv2d(256, 256, kernel_size = 3, padding = 'same')
    self.conv15 = torch.nn.Conv2d(256, 128, kernel_size = 3, padding = 'same')
    self.conv16 = torch.nn.Conv2d(128, 128, kernel_size = 3, padding = 'same')
    self.conv17 = torch.nn.Conv2d(128, 64, kernel_size = 3, padding = 'same')
    self.conv18 = torch.nn.Conv2d(64, 64, kernel_size = 3, padding = 'same')
    self.conv19 = torch.nn.Conv2d(64, 1, kernel_size=3, padding='same')

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

    x = self.maxpool(x) # output (128, 32, 32)

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

    x = self.maxpool(x) # output (512, 8, 8)

    x = self.conv9(x) # output (1024, 8, 8)
    x = self.relu(x)
    x = self.conv10(x) # output (1024, 8, 8)
    x = self.relu(x)

    # Decode

    x = self.upConv1(x) # output (512, 16, 16)
    x = torch.cat((copy4, x), dim = 1) # output (1024, 16, 16)

    x = self.conv11(x) # output (512, 16, 16)
    x = self.relu(x)
    x = self.conv12(x) # output (512, 16, 16)
    x = self.relu(x)

    x = self.upConv2(x) # output (256, 32, 32)
    x = torch.cat((copy3, x), dim = 1) # output (512, 32, 32)

    x = self.conv13(x) # output (256, 32, 32)
    x = self.relu(x)
    x = self.conv14(x) # output (256, 32, 32)
    x = self.relu(x)

    x = self.upConv3(x) # output (128, 64, 64)
    x = torch.cat((copy2, x), dim = 1) # output (256, 64, 64)

    x = self.conv15(x) # output (128, 64, 64)
    x = self.relu(x)
    x = self.conv16(x) # output (128, 64, 64)
    x = self.relu(x)

    x = self.upConv4(x) # output (64, 128, 128)
    x = torch.cat((copy1, x), dim = 1) # output (128, 128, 128)

    x = self.conv17(x) # output (64, 128, 128)
    x = self.relu(x)
    x = self.conv18(x) # output (64, 128, 128)
    x = self.relu(x)

    x = self.conv19(x) # output (1, 128, 128)

    return x

#%%

# Model Training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=.0002)
loss_fun = torch.nn.BCEWithLogitsLoss()
sigmoid = torch.nn.Sigmoid()

num_epochs = 45
ACE_train = []
ACE_val = []

for ep in range(num_epochs):
    print(f'Ep: {ep}')

    # Optimize parameters
    for x_batch, y_batch in dataloader_train:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        loss = loss_fun(outputs, y_batch)
        model.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        
        # Evaluation - Training data
        ace = 0
        for x_batch, y_batch in dataloader_train:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = loss_fun(outputs, y_batch)
            ace += loss.item() * x_batch.size(0)

        ACE = ace / len(dataloader_train.dataset)
        ACE_train.append(ACE)
        print(f'Average cross-entropy (training): {ACE}')

        # Evaluation - Validation data
        ace = 0
        for x_batch, y_batch in dataloader_val:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = loss_fun(outputs, y_batch)
            ace += loss.item() * x_batch.size(0)

        ACE = ace / len(dataloader_val.dataset)
        ACE_val.append(ACE)
        print(f'Average cross-entropy (validation): {ACE}')
        
#%%

# Convergence Plots

plt.figure()
plt.plot(ACE_train, label = "Training")
plt.plot(ACE_val, label = "Validation")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Binary Cross Entropy vs. Epoch")
plt.show()

#%%

# Test Data

counts = []

for x_batch in dataloader_test:
    x_batch = x_batch.to(device)
    outputs = model(x_batch) # Calculate predictions
    outputs = sigmoid(outputs)
    outputs_rounded = torch.round(outputs) # Round probabilities

    for i in range(outputs_rounded.shape[0]):
        img = outputs_rounded[i]
        img = img.squeeze().cpu().detach().numpy() # Coerce object type
        img = img.astype(np.uint8)

        info = cv2.connectedComponents(img)
        num_components = info[0]
        counts.append(num_components)
        
#%%  

# Submission

dct = {'index': [x for x in range(len(counts))], 'count': counts}
df_submission = pd.DataFrame(dct)
df_submission.to_csv('./FinalProject_submission.csv', index = False)
