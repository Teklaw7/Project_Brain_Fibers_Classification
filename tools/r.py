import numpy as np
import pandas as pd
import shutil, time, os, requests, random, copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.manifold import TSNE
def set_seed(seed = 16):
    np.random.seed(seed)
    torch.manual_seed(seed)

