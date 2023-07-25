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

lights = pd.read_pickle(r'lights_57_128d_on_sphere.pickle')
liste = os.listdir("/CMF/data/timtey/results_contrastive_learning_062623")
l_colors = colors.ListedColormap ( np.random.rand (57,3))
l_colors1 = colors.ListedColormap ( np.random.rand (20,3))
l_colors2 = colors.ListedColormap ( np.random.rand (20,3))
l_colors3 = colors.ListedColormap ( np.random.rand (17,3))

lights = torch.tensor(lights)
matrix2 = [] #should be shape = (56*100,128)

for i in range(len(liste)):
    matrix = torch.load(f"/CMF/data/timtey/results_contrastive_learning_062623/{liste[i]}")
    matrix2.append(matrix)
MATR2 = torch.cat(matrix2, dim=0)
Data_lab = MATR2[:,-1]
d_unique = torch.unique(Data_lab)
LAB = MATR2[:,-3]
MATR = MATR2[:,:128]
MATR = MATR.cpu()
LAB = LAB.cpu()
LIGHTS = lights.cpu()

threedtsne = PCA(n_components=3)
twopca = PCA(n_components=2)

GLOBAL = torch.cat((MATR, LIGHTS), dim=0)

uniq_lab = torch.unique(LAB)
threedtsne_results_global = threedtsne.fit_transform(GLOBAL)
results = twopca.fit_transform(GLOBAL)
threedtsne_results = threedtsne_results_global[:MATR.shape[0],:]
threedtsne_results_lights = threedtsne_results_global[MATR.shape[0]:,:]
pca_res = results[:MATR.shape[0],:]
pca_lights = results[MATR.shape[0]:,:]

plt.scatter(pca_res[:,0], pca_res[:,1], c=LAB, cmap=l_colors)
plt.scatter(pca_lights[:,0], pca_lights[:,1], c="black", marker="x")
plt.show()

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
    ax= plt.axes(projection='3d')
    ax.text(threedtsne_results_lights[i, 0], threedtsne_results_lights[i, 1], threedtsne_results_lights[i, 2], str(i), fontsize=12)
    ax.scatter(threedtsne_results_lights[i,0], threedtsne_results_lights[i,1], threedtsne_results_lights[i,2], linewidths=1, c='black')
    # ax.scatter(threedtsne_results[i*205:(i+1)*205,0], threedtsne_results[i*205:(i+1)*205,1], threedtsne_results[i*205:(i+1)*205,2], c=LAB[i*205:(i+1)*205], cmap=l_colors)
    ax.scatter(threedtsne_results[l_i,0], threedtsne_results[l_i,1], threedtsne_results[l_i,2], c=LAB[l_i], cmap=l_colors)

    ax.axes.set_xlim3d(left=-0.15, right=0.30)
    ax.axes.set_ylim3d(bottom=-0.2, top=0.4)
    ax.axes.set_zlim3d(bottom=-0.15, top=0.20)
    plt.show()