import numpy as np
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl 
import torchvision.models as models
from torch.nn.functional import softmax
import torchmetrics
from tools import utils
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)
from pytorch3d.renderer.blending import sigmoid_alpha_blend, hard_rgb_blend
from pytorch3d.structures import Meshes, join_meshes_as_scene

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pytorch3d.vis.plotly_vis import plot_scene
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from sklearn.utils.class_weight import compute_class_weight
import random
import pytorch3d.transforms as T3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
# import MLP
import random
import os
from sklearn import metrics
# import random
import matplotlib.colors as colors
import umap
import pandas as pd

lights = pd.read_pickle(r'lights_57_3d_on_positive_sphere.pickle')
# liste = os.listdir("/CMF/data/timtey/results_contrastive_loss_combine_loss_tract_cluster_bundle")
liste = os.listdir("/CMF/data/timtey/results_contrastive_learning_060923_3d")
l_colors = colors.ListedColormap ( np.random.rand (57,3))
l_colors1 = colors.ListedColormap ( np.random.rand (20,3))
l_colors2 = colors.ListedColormap ( np.random.rand (20,3))
l_colors3 = colors.ListedColormap ( np.random.rand (17,3))

lights = torch.tensor(lights)
print(lights.shape)
matrix2 = [] #should be shape = (56*100,128)

for i in range(len(liste)):
    matrix = torch.load(f"/CMF/data/timtey/results_contrastive_learning_060923_3d/{liste[i]}")
    matrix2.append(matrix)
MATR2 = torch.cat(matrix2, dim=0)
print(MATR2.shape)
Data_lab = MATR2[:,-1]
print(Data_lab.shape)
d_unique = torch.unique(Data_lab)
print(d_unique)
LAB = MATR2[:,-3]
print(LAB.shape)
print(LAB)
MATR = MATR2[:,:128]
MATR = MATR.cpu()
LAB = LAB.cpu()
LIGHTS = lights.cpu()

uniq_lab = torch.unique(LAB)

ax = plt.axes(projection='3d')
ax.scatter(LIGHTS[:,0], LIGHTS[:,1], LIGHTS[:,2], linewidths=5, c='black')
for i in range(LIGHTS.shape[0]):
    ax.text(LIGHTS[i, 0], LIGHTS[i, 1], LIGHTS[i, 2], str(i), fontsize=12)
plt.show()

ax = plt.axes(projection='3d')
ax.scatter(LIGHTS[:,0], LIGHTS[:,1], LIGHTS[:,2], linewidths=5, c='black')
ax.scatter(MATR[:,0], MATR[:,1], MATR[:,2], c=LAB, cmap=l_colors)
for i in range(LIGHTS.shape[0]):
    ax.text(LIGHTS[i, 0], LIGHTS[i, 1], LIGHTS[i, 2], str(i), fontsize=12)
ax.axes.set_xlim3d(left=0, right=1)
ax.axes.set_ylim3d(bottom=0, top=1)
ax.axes.set_zlim3d(bottom=0, top=1)
plt.show()

for i in range(LIGHTS.shape[0]):
    l_i = (LAB==i).nonzero().squeeze()
    l_i = [j.item() for j in l_i]
    # print(l_i)
    # print(LAB)
    # print(LAB[l_i])
    # print(LAB[l_i].shape)
    # print(threedtsne_results[l_i,:].shape)
    ax= plt.axes(projection='3d')
    ax.text(LIGHTS[i, 0], LIGHTS[i, 1], LIGHTS[i, 2], str(i), fontsize=12)
    ax.scatter(LIGHTS[i,0], LIGHTS[i,1], LIGHTS[i,2], linewidths=1, c='black')
    # ax.scatter(threedtsne_results[i*205:(i+1)*205,0], threedtsne_results[i*205:(i+1)*205,1], threedtsne_results[i*205:(i+1)*205,2], c=LAB[i*205:(i+1)*205], cmap=l_colors)
    ax.scatter(MATR[l_i,0], MATR[l_i,1], MATR[l_i,2], c=LAB[l_i], cmap=l_colors)

    ax.axes.set_xlim3d(left=0, right=1)
    ax.axes.set_ylim3d(bottom=0, top=1)
    ax.axes.set_zlim3d(bottom=0, top=1)
    plt.show()
    # print(LAB[i*205:(i+1)*205])