# Imports
from __future__ import print_function, division

import time
import numpy as np
import os 
import copy

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from torch.optim import lr_scheduler
from PIL import Image
import argparse

# Setting argparse module
parser = argparse.ArgumentParser(description='Train an image classifier')

parser.add_argument('--data_directory',default='flowers', type=str, help='dataset files')
parser.add_argument('--gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--arch', type=str, default ='vgg13' help='model architecture')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='hidden units')
parser.add_argument('--epochs', type=int, default=20,  help='epochs')
parser.add_argument('--save_dir', type=str, default = 'checkpoint.pth', help = 'saving checkpoint')