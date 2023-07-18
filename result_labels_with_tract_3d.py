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
import math as m
import statistics
import pickle




lights = pd.read_pickle(r'lights_57_3d_on_sphere.pickle')
# liste = os.listdir("/CMF/data/timtey/results_contrastive_loss_combine_loss_tract_cluster_bundle")
# liste = os.listdir("/CMF/data/timtey/results_contrastive_learning_063023_best_model")
liste = os.listdir("/CMF/data/timtey/results_contrastive_learning_071023_best")
l_colors = colors.ListedColormap ( np.random.rand (57,3))
l_colors1 = colors.ListedColormap ( np.random.rand (20,3))
l_colors2 = colors.ListedColormap ( np.random.rand (20,3))
l_colors3 = colors.ListedColormap ( np.random.rand (17,3))

lights = torch.tensor(lights)
print(lights.shape)
matrix2 = [] #should be shape = (56*100,128)
matrix_1 = []
matrix_2 = []
for i in range(len(liste)):
    # matrix = torch.load(f"/CMF/data/timtey/results_contrastive_learning_063023_best_model/{liste[i]}")
    matrix = torch.load(f"/CMF/data/timtey/results_contrastive_learning_071023_best/{liste[i]}")
    matrix_bundle = matrix[:matrix.shape[0]//2]
    matrix_tract = matrix[matrix.shape[0]//2:]
    # if i==0:
        # print(matrix.shape)
        # print(matrix[:,-1])
    # for j in range(matrix.shape[0]):
    #     if j <=matrix.shape[0]//2:
    #         matrix_1.append(matrix[j,:])
    #     else:
    #         matrix_2.append(matrix[j,:])
    # if i ==0:
    #     print(torch.tensor(matrix_1).shape)
    #     print(torch.tensor(matrix_2).shape)
    matrix2.append(matrix)
    matrix_1.append(matrix_bundle)
    matrix_2.append(matrix_tract)
MATR2 = torch.cat(matrix2, dim=0)
print(MATR2.shape)
MATR_1 = torch.cat(matrix_1, dim=0)
print(MATR_1.shape)
MATR_2 = torch.cat(matrix_2, dim=0)
print(MATR_2.shape)
print(MATR_1[0])
MATRB = MATR_1[:,:3]
MATRT = MATR_2[:,:3]
LABb = MATR_1[:,3]
LABt = MATR_2[:,3]
print(MATRB.shape, MATRB[0])
print(MATRT.shape)
print(LABb.shape, LABb[0])
print(LABt.shape)

n_labB = MATR_1[:,4:7]
n_labT = MATR_2[:,4:7]
print(n_labB.shape, n_labB[0])
print(n_labT.shape)
Data_labB = MATR_1[:,-1]
Data_labT = MATR_2[:,-1]
print(Data_labB.shape, Data_labB[0])
print(Data_labT.shape)




Mean_pos = torch.tensor([])
Mean_pos = Mean_pos.to('cpu')
for j in range(lights.shape[0]):
    L_j = []
    for i in range(MATRB.shape[0]):
        if int(LABb[i].item())==j:
            L_j.append(i)
    MATRb_i = MATRB[L_j]
    Mean_pos_j = torch.mean(MATRb_i, dim=0)
    Mean_pos_j = Mean_pos_j/np.sqrt(Mean_pos_j[0]**2 + Mean_pos_j[1]**2 + Mean_pos_j[2]**2)
    Mean_pos_j = Mean_pos_j.to('cpu')
    Mean_pos = torch.cat((Mean_pos, Mean_pos_j.unsqueeze(0)), dim=0)

# print(Mean_pos.shape)

# print(LABt)
# L_accept = []
# for i in range(MATRT.shape[0]):
    # print(i)
    # distance = torch.tensor([])
    # for j in range(Mean_pos.shape[0]):
        # dist = 1 * np.arccos(MATRT[i][0]*Mean_pos[j][0] + MATRT[i][1]*Mean_pos[j][1] + MATRT[i][2]*Mean_pos[j][2])
        # dist = torch.tensor(dist)
        # distance = torch.cat((distance, dist.unsqueeze(0)), dim=0)
    # min_dist = torch.argmin(distance).item()
    # print(torch.min(distance).item())
    # LABt[i] = min_dist
    # if torch.min(distance).item() < 0.1:
        # L_accept.append(i)

# 
# print(LABt)
# torch.save(LABt, "LABt_071023_best_mean_pos.pt")
LABt = torch.load("LABt_071023_best_mean_pos.pt")

# len(L_accept)
# file_name = "to_accept_01.pkl"
# open_file = open(file_name, "wb")
# pickle.dump(L_accept, open_file)
# open_file.close()


lights = lights.cpu()
MATRB = MATRB.cpu()
LABb = LABb.cpu()
n_labB = n_labB.cpu()
Data_labB = Data_labB.cpu()
MATRT = MATRT.cpu()
LABt = LABt.cpu()
n_labT = n_labT.cpu()
Data_labT = Data_labT.cpu()


# MATRB = MATRB[L_accept]
# LABb = LABb[L_accept]
# n_labB = n_labB[L_accept]
# Data_labB = Data_labB[L_accept]


openfile = open("to_accept_01.pkl", "rb")
L_accept = pickle.load(openfile)
openfile.close()

MATRT = MATRT[L_accept]
LABt = LABt[L_accept]
n_labT = n_labT[L_accept]
Data_labT = Data_labT[L_accept]




ax = plt.axes(projection='3d')
ax.scatter(lights[:,0], lights[:,1], lights[:,2], linewidths=5, c='black')
for i in range(lights.shape[0]):
    ax.text(lights[i, 0], lights[i, 1], lights[i, 2], str(i), fontsize=12)
ax.scatter(Mean_pos[:,0], Mean_pos[:,1], Mean_pos[:,2], c='red')
for j in range(Mean_pos.shape[0]):
    ax.text(Mean_pos[j, 0], Mean_pos[j, 1], Mean_pos[j, 2], str(j), fontsize=12)
# ax.view_init(90, 90,45)
plt.show()

ax = plt.axes(projection='3d')
ax.scatter(lights[:,0], lights[:,1], lights[:,2], linewidths=5, c='black')
ax.scatter(MATRB[:,0], MATRB[:,1], MATRB[:,2], c=LABb, cmap=l_colors)
for i in range(lights.shape[0]):
    ax.text(lights[i, 0], lights[i, 1], lights[i, 2], str(i), fontsize=12)
ax.scatter(Mean_pos[:,0], Mean_pos[:,1], Mean_pos[:,2], c='red')
for j in range(Mean_pos.shape[0]):
    ax.text(Mean_pos[j, 0], Mean_pos[j, 1], Mean_pos[j, 2], str(j), fontsize=12)
# ax.axes.set_xlim3d(left=0, right=1)
# ax.axes.set_ylim3d(bottom=0, top=1)
# ax.axes.set_zlim3d(bottom=0, top=1)
ax.axes.set_xlim3d(left=-1, right=1)
ax.axes.set_ylim3d(bottom=-1, top=1)
ax.axes.set_zlim3d(bottom=-1, top=1)
plt.show()


ax = plt.axes(projection='3d')
ax.scatter(MATRT[:,0], MATRT[:,1], MATRT[:,2], marker='*', c=LABt, cmap=l_colors)
ax.scatter(lights[:,0], lights[:,1], lights[:,2], linewidths=5, c='black')
ax.scatter(MATRB[:,0], MATRB[:,1], MATRB[:,2], c=LABb, cmap=l_colors)
for i in range(lights.shape[0]):
    ax.text(lights[i, 0], lights[i, 1], lights[i, 2], str(i), fontsize=12)
# ax.axes.set_xlim3d(left=0, right=1)
# ax.axes.set_ylim3d(bottom=0, top=1)
# ax.axes.set_zlim3d(bottom=0, top=1)
ax.axes.set_xlim3d(left=-1, right=1)
ax.axes.set_ylim3d(bottom=-1, top=1)
ax.axes.set_zlim3d(bottom=-1, top=1)
plt.show()


for i in range(lights.shape[0]):
    l_i = (LABb==i).nonzero().squeeze()
    l_i = [j.item() for j in l_i]
    # print(l_i)
    # print(LAB)
    # print(LAB[l_i])
    # print(LAB[l_i].shape)
    # print(threedtsne_results[l_i,:].shape)
    ax= plt.axes(projection='3d')
    ax.scatter(lights[:,0], lights[:,1], lights[:,2], linewidths=1, c='blue')
    ax.text(lights[i, 0], lights[i, 1], lights[i, 2], str(i), fontsize=12)
    ax.scatter(lights[i,0], lights[i,1], lights[i,2], linewidths=1, c='black')
    # ax.scatter(threedtsne_results[i*205:(i+1)*205,0], threedtsne_results[i*205:(i+1)*205,1], threedtsne_results[i*205:(i+1)*205,2], c=LAB[i*205:(i+1)*205], cmap=l_colors)
    ax.scatter(MATRB[l_i,0], MATRB[l_i,1], MATRB[l_i,2], c=LABb[l_i], cmap=l_colors)
    ax.scatter(Mean_pos[i,0], Mean_pos[i,1], Mean_pos[i,2], c='green')
    # ax.axes.set_xlim3d(left=0, right=1)
    # ax.axes.set_ylim3d(bottom=0, top=1)
    # ax.axes.set_zlim3d(bottom=0, top=1)
    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=-1, top=1)
    plt.show()
    # print(LAB[i*205:(i+1)*205])