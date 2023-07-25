import numpy as np
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os
import matplotlib.colors as colors
import pandas as pd

# lights = pd.read_pickle(r'lights_57_3d_on_positive_sphere.pickle')
lights = pd.read_pickle(r'lights_57_3d_on_sphere.pickle')
# liste = os.listdir("/CMF/data/timtey/results_contrastive_loss_combine_loss_tract_cluster_bundle")
liste = os.listdir("/CMF/data/timtey/results_contrastive_learning_062723")
l_colors = colors.ListedColormap ( np.random.rand (57,3))
l_colors1 = colors.ListedColormap ( np.random.rand (20,3))
l_colors2 = colors.ListedColormap ( np.random.rand (20,3))
l_colors3 = colors.ListedColormap ( np.random.rand (17,3))

lights = torch.tensor(lights)
matrix2 = [] #should be shape = (56*100,128)

for i in range(len(liste)):
    matrix = torch.load(f"/CMF/data/timtey/results_contrastive_learning_062723/{liste[i]}")
    # matrix = torch.load(f"/CMF/data/timtey/results_contrastive_learning_061523/{liste[i]}")
    matrix2.append(matrix)
MATR2 = torch.cat(matrix2, dim=0)
Data_lab = MATR2[:,-1]
d_unique = torch.unique(Data_lab)
LAB = MATR2[:,-3]
MATR = MATR2[:,:3]
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

ax.axes.set_xlim3d(left=-1, right=1)
ax.axes.set_ylim3d(bottom=-1, top=1)
ax.axes.set_zlim3d(bottom=-1, top=1)
plt.show()

for i in range(LIGHTS.shape[0]):
    l_i = (LAB==i).nonzero().squeeze()
    l_i = [j.item() for j in l_i]
    ax= plt.axes(projection='3d')
    ax.scatter(LIGHTS[:,0], LIGHTS[:,1], LIGHTS[:,2], linewidths=1, c='blue')
    ax.text(LIGHTS[i, 0], LIGHTS[i, 1], LIGHTS[i, 2], str(i), fontsize=12)
    ax.scatter(LIGHTS[i,0], LIGHTS[i,1], LIGHTS[i,2], linewidths=1, c='black')
    # ax.scatter(threedtsne_results[i*205:(i+1)*205,0], threedtsne_results[i*205:(i+1)*205,1], threedtsne_results[i*205:(i+1)*205,2], c=LAB[i*205:(i+1)*205], cmap=l_colors)
    ax.scatter(MATR[l_i,0], MATR[l_i,1], MATR[l_i,2], c=LAB[l_i], cmap=l_colors)
    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=-1, top=1)
    plt.show()