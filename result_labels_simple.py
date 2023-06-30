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

lights = pd.read_pickle(r'lights_good_on_sphere.pickle')
# liste = os.listdir("/CMF/data/timtey/results_contrastive_loss_combine_loss_tract_cluster_bundle")
liste = os.listdir("/CMF/data/timtey/results_contrastive_learning_062623")
l_colors = colors.ListedColormap ( np.random.rand (57,3))
l_colors1 = colors.ListedColormap ( np.random.rand (20,3))
l_colors2 = colors.ListedColormap ( np.random.rand (20,3))
l_colors3 = colors.ListedColormap ( np.random.rand (17,3))

lights = torch.tensor(lights)
print(lights.shape)
matrix2 = [] #should be shape = (56*100,128)

for i in range(len(liste)):
    matrix = torch.load(f"/CMF/data/timtey/results_contrastive_learning_062623/{liste[i]}")
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


# for i in range(LIGHTS.shape[0]):
    # print(np.linalg.norm(LIGHTS[i]))
# for i in range(MATR.shape[0]):
    # print(np.linalg.norm(MATR[i]))
    # MATR[i] = MATR[i]/np.linalg.norm(MATR[i])
# print(np.linalg.norm(LIGHTS, axis=1, keepdims=True))
# print(np.linalg.norm(MATR, axis=1, keepdims=True))
# print(LIGHTS[0],)
# threedtsne = TSNE(n_components=3, perplexity=205)
threedtsne = PCA(n_components=3)
twopca = PCA(n_components=2)
# tsne = PCA(n_components=2)
# tsne = TSNE(n_components=2, perplexity=500)
# lightstest = threedtsne.fit_transform(LIGHTS)
# ax = plt.axes(projection='3d')
# ax.scatter(lightstest[:,0], lightstest[:,1], lightstest[:,2])
# ax.axes.set_xlim3d(left=-1, right=1)
# ax.axes.set_ylim3d(bottom=-1, top=1)
# ax.axes.set_zlim3d(bottom=-1, top=1)
# plt.show()
# lightstest = np.abs(lightstest/np.linalg.norm(lightstest,axis=1,keepdims=True))
# ax = plt.axes(projection='3d')
# ax.scatter(lightstest[:,0], lightstest[:,1], lightstest[:,2])
# ax.axes.set_xlim3d(left=-1, right=1)
# ax.axes.set_ylim3d(bottom=-1, top=1)
# ax.axes.set_zlim3d(bottom=-1, top=1)
# plt.show()
# print(djdshkjfhdjk)
# threedumap = umap.UMAP(n_components=3, n_neighbors=205, min_dist=0.1)
# twodumap = umap.UMAP(n_components=2, n_neighbors=205, min_dist=0.1)
GLOBAL = torch.cat((MATR, LIGHTS), dim=0)
print("GLOBAL",GLOBAL.shape)
# tsne_results = tsne.fit_transform(MATR)
# tsne_results_lights = tsne.fit_transform(LIGHTS)
# tsne_results_global = tsne.fit_transform(GLOBAL)
# tsne_results_global = twodumap.fit_transform(GLOBAL)
# tsne_results = tsne_results_global[:MATR.shape[0],:]
# tsne_results_lights = tsne_results_global[MATR.shape[0]:,:]
# torch.save(tsne_results, "tsne_results_contrastive_learning_060523_2_before_normalisation.pt")
# torch.save(tsne_results_lights, "tsne_results_lights_contrastive_learning_060523_2_before_normalisation.pt")
# umap_results = twodumap.fit_transform(MATR)
# umap_results_lights = twodumap.fit_transform(LIGHTS)
uniq_lab = torch.unique(LAB)
# threedumap_results = threedumap.fit_transform(MATR)
# threedumap_results_lights = threedumap.fit_transform(LIGHTS)
# threedtsne_results = threedtsne.fit_transform(MATR)
# threedtsne_results_lights = threedtsne.fit_transform(LIGHTS)
threedtsne_results_global = threedtsne.fit_transform(GLOBAL)
results = twopca.fit_transform(GLOBAL)
# threedtsne_results_global = threedumap.fit_transform(GLOBAL)
threedtsne_results = threedtsne_results_global[:MATR.shape[0],:]
threedtsne_results_lights = threedtsne_results_global[MATR.shape[0]:,:]
pca_res = results[:MATR.shape[0],:]
pca_lights = results[MATR.shape[0]:,:]

plt.scatter(pca_res[:,0], pca_res[:,1], c=LAB, cmap=l_colors)
plt.scatter(pca_lights[:,0], pca_lights[:,1], c="black", marker="x")
plt.show()
# torch.save(threedtsne_results, "threedtsne_results_contrastive_learning_052523_before_normalisation.pt")
# torch.save(threedtsne_results_lights, "threedtsne_results_lights_contrastive_learning_052523_before_normalisation.pt")
# torch.save(tsne_results, "tsne_results_contrastive_learning_052523_before_normalisation.pt")
# torch.save(tsne_results_lights, "tsne_results_lights_contrastive_learning_052523_before_normalisation.pt")
# tsne_results = torch.load("pt/053023/tsne_results_contrastive_learning_053023_before_normalisation.pt")
# tsne_results_lights = torch.load("pt/053023/tsne_results_lights_contrastive_learning_053023_before_normalisation.pt")
# plt.scatter(tsne_results[:,0], tsne_results[:,1], c=LAB, cmap=l_colors)
# plt.scatter(tsne_results_lights[:,0], tsne_results_lights[:,1])
# for i in range(len(tsne_results_lights)):
    # plt.text(tsne_results_lights[i, 0], tsne_results_lights[i, 1], str(i), fontsize=12)
# plt.colorbar(ticks = uniq_lab)
# plt.show()

# threedtsne_results = np.abs(threedtsne_results/np.linalg.norm(threedtsne_results, axis=1, keepdims=True))
# threedtsne_results_lights = np.abs(threedtsne_results_lights/np.linalg.norm(threedtsne_results_lights, axis=1, keepdims=True))
# threedtsne_results = threedtsne_results/np.linalg.norm(threedtsne_results, axis=1, keepdims=True)
# threedtsne_results_lights = threedtsne_results_lights/np.linalg.norm(threedtsne_results_lights, axis=1, keepdims=True)

# torch.save(threedtsne_results, "threedtsne_results_contrastive_learning_060823.pt")
# torch.save(threedtsne_results_lights, "threedtsne_results_lights_contrastive_learning_060823.pt")
# threedtsne_results = torch.load("pt/053023/threedtsne_results_contrastive_learning_053023.pt")
# threedtsne_results_lights = torch.load("pt/053023/threedtsne_results_lights_contrastive_learning_053023.pt")
# ax = plt.axes(projection='3d')
# ax.scatter(threedtsne_results[:,0], threedtsne_results[:,1], threedtsne_results[:,2], c=LAB, cmap=l_colors)
# plt.show()

ax = plt.axes(projection='3d')
ax.scatter(threedtsne_results_lights[:,0], threedtsne_results_lights[:,1], threedtsne_results_lights[:,2], linewidths=5, c='black')
ax.scatter(threedtsne_results[:,0], threedtsne_results[:,1], threedtsne_results[:,2], c=LAB, cmap=l_colors)
for i in range(len(threedtsne_results_lights)):
    ax.text(threedtsne_results_lights[i, 0], threedtsne_results_lights[i, 1], threedtsne_results_lights[i, 2], str(i), fontsize=12)
# ax.axes.set_xlim3d(left=-1, right=1)
# ax.axes.set_ylim3d(bottom=-1, top=1)
# ax.axes.set_zlim3d(bottom=-1, top=1)
plt.show()

for i in range(len(threedtsne_results_lights)):
    l_i = (LAB==i).nonzero().squeeze()
    l_i = [j.item() for j in l_i]
    # print(l_i)
    # print(LAB)
    # print(LAB[l_i])
    # print(LAB[l_i].shape)
    # print(threedtsne_results[l_i,:].shape)
    ax= plt.axes(projection='3d')
    ax.text(threedtsne_results_lights[i, 0], threedtsne_results_lights[i, 1], threedtsne_results_lights[i, 2], str(i), fontsize=12)
    ax.scatter(threedtsne_results_lights[i,0], threedtsne_results_lights[i,1], threedtsne_results_lights[i,2], linewidths=1, c='black')
    # ax.scatter(threedtsne_results[i*205:(i+1)*205,0], threedtsne_results[i*205:(i+1)*205,1], threedtsne_results[i*205:(i+1)*205,2], c=LAB[i*205:(i+1)*205], cmap=l_colors)
    ax.scatter(threedtsne_results[l_i,0], threedtsne_results[l_i,1], threedtsne_results[l_i,2], c=LAB[l_i], cmap=l_colors)

    ax.axes.set_xlim3d(left=-0.15, right=0.30)
    ax.axes.set_ylim3d(bottom=-0.2, top=0.4)
    ax.axes.set_zlim3d(bottom=-0.15, top=0.20)
    plt.show()
    # print(LAB[i*205:(i+1)*205])