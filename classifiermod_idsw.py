# model code format transfer

# imports
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.nn import functional
from zipfile import ZipFile
import cv2
#import tensorflow as tf
import pickle
import seaborn as sns

from image_data_from_box import set_label_image_lists


def get_data(data_file_path):
  label_arr, image_arr = set_label_image_lists(data_file_path)
  random_seed = 42

  x_train, x_test, y_train, y_test = train_test_split(image_arr, label_arr, test_size=0.3,
                                                      random_state=42) #, stratify=np.array(label_arr)

  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size = 0.1, train_size=0.9,
                                        random_state=random_seed, shuffle = True) #,stratify=np.array(label_arr)[y_train]
  return x_train, y_train, x_test, y_test, x_val, y_val

embed_list = []

class ShapePrint(nn.Module):
  def __init__(self):
    super(ShapePrint, self).__init__()
  def forward(self, x):
    print(x.shape)
    return x

class xPrint(nn.Module):
  def __init__(self):
    super(xPrint, self).__init__()
  def forward(self, x):
    print(x)
    embed_list.append(x)
    return x

# ptrblck code for saving layer output from a sequential block
act_out = {}
def get_hook(name):
    def hook(m, input, output):
        act_out[name] = output.detach()
    return hook


class TorchNet(nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.flatten = nn.Flatten()

        self.conv_layers = nn.Sequential(  #
              nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2),
              nn.ReLU(),
              nn.Dropout(p=0.5),
              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2),
              nn.ReLU(),
              nn.Conv2d(in_channels =64, out_channels=64, kernel_size=3),
              nn.ReLU(),
              nn.MaxPool2d(2, 2),
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2),
              nn.ReLU(),
              nn.Conv2d(in_channels =128, out_channels=128, kernel_size=3),
              nn.ReLU(),
              nn.MaxPool2d(2,2),
              nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2),
              nn.ReLU(),
              nn.Conv2d(in_channels =256, out_channels=256, kernel_size=3),
              nn.ReLU(),
              nn.MaxPool2d(2,2),
              nn.Dropout(p=0.5),
          )

        self.linear_1 = nn.Sequential(
            nn.Linear(16384, 100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            xPrint(),
            nn.Dropout(p=0.5),
            nn.Linear(100,11),
            nn.Softmax(),

        )

    def forward(self, x):
      x= self.conv_layers(x)
      x = x.flatten()
      x = x.squeeze()
      x = self.linear_1(x)
      return x





class vgg16TorchNet(nn.Module):
    def __init__(self):
        super(vgg16TorchNet, self).__init__()
        self.flatten = nn.Flatten()

        self.conv_layers = nn.Sequential(  #
              nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2),
              nn.ReLU(), #inplace=True
              nn.Dropout(p=0.5),
              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2),
              nn.ReLU(), #inplace=True
              nn.Conv2d(in_channels =64, out_channels=64, kernel_size=3),
              nn.ReLU(),
              nn.MaxPool2d(2, 2),
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2),
              nn.ReLU(), #inplace=True
              nn.Conv2d(in_channels =128, out_channels=128, kernel_size=3),
              nn.ReLU(),
              nn.MaxPool2d(2,2),
              nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2),
              nn.ReLU(), #inplace=True
              nn.Conv2d(in_channels =256, out_channels=256, kernel_size=3),
              nn.ReLU(),
              nn.MaxPool2d(2,2),
              nn.Dropout(p=0.5), # [256, 128, 3, 3], expected input[1, 64, 9, 9]
          )

        self.linear_1 = nn.Sequential(    #1x16384 and 4096x100)
            nn.Linear(16384, 100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(100,11),
            nn.Softmax(),

        )

    def forward(self, x):
      #forward method. opposition to backward pass
      #print(x.shape)
      #print('in: ',x.shape)
      x= self.conv_layers(x)
      #print('post conv:  ',x.shape)
      x = x.flatten()
      x = x.squeeze()
      #print('pre flatten:', x.shape)
      #print('conv x', x.shape)
      x = self.linear_1(x)
      #print('out: ',x.shape)
      #print('lin1 x', x)
      return x


