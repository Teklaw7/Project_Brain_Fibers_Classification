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
import pickle

lights = pd.read_pickle(r'lights_57_3d_on_sphere.pickle')
# liste = os.listdir("/CMF/data/timtey/results_contrastive_loss_combine_loss_tract_cluster_bundle")
# liste = os.listdir("/CMF/data/timtey/results_contrastive_learning_063023_best_model")
liste = os.listdir("/CMF/data/timtey/results_contrastive_learning_071023")
l_colors = colors.ListedColormap ( np.random.rand (57,3))
l_colors1 = colors.ListedColormap ( np.random.rand (20,3))
l_colors2 = colors.ListedColormap ( np.random.rand (20,3))
l_colors3 = colors.ListedColormap ( np.random.rand (17,3))

lights = torch.tensor(lights)
matrix2 = [] #should be shape = (56*100,128)
matrix_1 = []
matrix_2 = []
for i in range(len(liste)):
    # matrix = torch.load(f"/CMF/data/timtey/results_contrastive_learning_063023_best_model/{liste[i]}")
    matrix = torch.load(f"/CMF/data/timtey/results_contrastive_learning_071023/{liste[i]}")
    matrix_bundle = matrix[:matrix.shape[0]//2]
    matrix_tract = matrix[matrix.shape[0]//2:]
    matrix2.append(matrix)
    matrix_1.append(matrix_bundle)
    matrix_2.append(matrix_tract)
MATR2 = torch.cat(matrix2, dim=0)
MATR_1 = torch.cat(matrix_1, dim=0)
MATR_2 = torch.cat(matrix_2, dim=0)
MATRB = MATR_1[:,:3]
MATRT = MATR_2[:,:3]
LABb = MATR_1[:,3]
LABt = MATR_2[:,3]
n_labB = MATR_1[:,4:7]
n_labT = MATR_2[:,4:7]
Data_labB = MATR_1[:,-1]
Data_labT = MATR_2[:,-1]

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

L_accept = []
for i in range(MATRT.shape[0]):
    print(i)
    distance = torch.tensor([])
    for j in range(Mean_pos.shape[0]):
        dist = 1 * np.arccos(MATRT[i][0]*Mean_pos[j][0] + MATRT[i][1]*Mean_pos[j][1] + MATRT[i][2]*Mean_pos[j][2])
        dist = torch.tensor(dist)
        distance = torch.cat((distance, dist.unsqueeze(0)), dim=0)
    min_dist = torch.argmin(distance).item()
    LABt[i] = min_dist
    if torch.min(distance).item() < 0.1:
        L_accept.append(i)

torch.save(LABt, "LABt_071023_mean_pos.pt")
# LABt = torch.load("LABt_071023_best_mean_pos.pt")

file_name = "to_accept_01_071023.pkl"
open_file = open(file_name, "wb")
pickle.dump(L_accept, open_file)
open_file.close()


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


# openfile = open("to_accept_01.pkl", "rb")
# L_accept = pickle.load(openfile)
# openfile.close()

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
plt.show()

ax = plt.axes(projection='3d')
ax.scatter(lights[:,0], lights[:,1], lights[:,2], linewidths=5, c='black')
ax.scatter(MATRB[:,0], MATRB[:,1], MATRB[:,2], c=LABb, cmap=l_colors)
for i in range(lights.shape[0]):
    ax.text(lights[i, 0], lights[i, 1], lights[i, 2], str(i), fontsize=12)
ax.scatter(Mean_pos[:,0], Mean_pos[:,1], Mean_pos[:,2], c='red')
for j in range(Mean_pos.shape[0]):
    ax.text(Mean_pos[j, 0], Mean_pos[j, 1], Mean_pos[j, 2], str(j), fontsize=12)
ax.axes.set_xlim3d(left=-1, right=1)
ax.axes.set_ylim3d(bottom=-1, top=1)
ax.axes.set_zlim3d(bottom=-1, top=1)
plt.show()


ax = plt.axes(projection='3d')
ax.scatter(MATRT[:,0], MATRT[:,1], MATRT[:,2], marker='*', c=LABt, cmap=l_colors)
# ax.scatter(MATRT[:,0], MATRT[:,1], MATRT[:,2], marker='*')
ax.scatter(lights[:,0], lights[:,1], lights[:,2], linewidths=5, c='black')
ax.scatter(MATRB[:,0], MATRB[:,1], MATRB[:,2], c=LABb, cmap=l_colors)
for i in range(lights.shape[0]):
    ax.text(lights[i, 0], lights[i, 1], lights[i, 2], str(i), fontsize=12)
ax.axes.set_xlim3d(left=-1, right=1)
ax.axes.set_ylim3d(bottom=-1, top=1)
ax.axes.set_zlim3d(bottom=-1, top=1)
plt.show()


for i in range(lights.shape[0]):
    l_i = (LABb==i).nonzero().squeeze()
    l_i = [j.item() for j in l_i]
    ax= plt.axes(projection='3d')
    ax.scatter(lights[:,0], lights[:,1], lights[:,2], linewidths=1, c='blue')
    ax.text(lights[i, 0], lights[i, 1], lights[i, 2], str(i), fontsize=12)
    ax.scatter(lights[i,0], lights[i,1], lights[i,2], linewidths=1, c='black')
    # ax.scatter(threedtsne_results[i*205:(i+1)*205,0], threedtsne_results[i*205:(i+1)*205,1], threedtsne_results[i*205:(i+1)*205,2], c=LAB[i*205:(i+1)*205], cmap=l_colors)
    ax.scatter(MATRB[l_i,0], MATRB[l_i,1], MATRB[l_i,2], c=LABb[l_i], cmap=l_colors)
    ax.scatter(Mean_pos[i,0], Mean_pos[i,1], Mean_pos[i,2], c='green')
    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=-1, top=1)
    plt.show()