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



# def normalize(x):
#     for i in range(x.shape[0]):
#         norm =0
#         for j in range(x.shape[1]):
#             norm += x[i,j]**2
#         norm = np.sqrt(norm)
#         x[i] = x[i]/norm
#     return x




lights = pd.read_pickle(r'Lights_good.pickle')
# get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
# l_colors = get_colors(57) # sample return:  ['#8af5da', '#fbc08c', '#b741d0']
# l_colors = plt.colors.ListedColormap ( np.random.rand ( 256,57))
# l_colors = colorbar.Colormap()
# liste = os.listdir("/CMF/data/timtey/results_contrastive_learning/results2")
# liste = os.listdir("/CMF/data/timtey/results_contrastive_learning/results2_pretrained_true_t_04")
# liste = os.listdir("/CMF/data/timtey/results_contrastive_loss_combine")
liste = os.listdir("/CMF/data/timtey/results_contrastive_loss_combine_loss_tract_cluster_with_simclr")
# l_colors = ['#DF2F0A', '#F85904', '#FFA500', '#FFD700', '#FFFF00', '#ADFF2F', '#7FFF00', '#00FF00', '#00FA9A', '#00FFFF', '#00BFFF', '#1E90FF', '#0000FF', '#8A2BE2', '#FF00FF', '#FF1493', '#FFC0CB', '#FFA07A', '#FF4500', '#A52A2A', '#808080', '#000000', '#FFFFFF', '#FFDAB9', '#FFE4B5', '#F0E68C', '#EEE8AA', '#BDB76B', '#556B2F', '#006400', '#008000', '#228B22', '#2E8B57', '#3CB371', '#20B2AA', '#00FFFF', '#00CED1', '#1E90FF', '#0000FF', '#000080', '#483D8B', '#8A2BE2', '#9932CC', '#FF00FF', '#FF1493', '#C71585', '#FFC0CB', '#FFA07A', '#FF4500', '#A52A2A', '#808080', '#000000', '#FFFFFF', '#FFDAB9', '#FFE4B5', '#F0E68C', '#EEE8AA', '#BDB76B', '#556B2F', '#006400', '#008000', '#228B22', '#2E8B57', '#3CB371', '#20B2AA', '#00FFFF', '#00CED1', '#1E90FF', '#0000FF', '#000080', '#483D8B', '#8A2BE2', '#9932CC', '#FF00FF', '#FF1493', '#C71585', '#FFC0CB', '#FFA07A', '#FF4500', '#A52A2A', '#808080', '#000000', '#FFFFFF', '#FFDAB9', '#FFE4B5', '#F0E68C', '#EEE8AA', '#BDB76B', '#556B2F', '#006400', '#008000', '#228B22', '#2E8B57', '#3CB371', '#20B2AA', '#00FFFF', '#00CED1', '#1E90FF', '#0000FF', '#000080', '#483D8B', '#8A2BE2', '#9932CC', '#FF00FF', '#FF1493', '#C71585', '#FFC0CB', '#FFA07A', '#FF4500', '#A52A2A', '#808080', '#000000', '#FFFFFF', '#FFDAB9', '#FFE4B5', '#F0E68C', '#EEE8AA']
np.random.seed(125467)
l_colors = colors.ListedColormap ( np.random.rand (57,3))
l_colors1 = colors.ListedColormap ( np.random.rand (20,3))
l_colors2 = colors.ListedColormap ( np.random.rand (20,3))
l_colors3 = colors.ListedColormap ( np.random.rand (17,3))
l_colors11 = colors.ListedColormap ( np.random.rand (1,3))
# colors.ListedColormap.
# L_c = []
# L_c.append(l_colors)
# tensor_l_colors = torch.tensor(L_c)
# torch.save(tensor_l_colors, "/home/timtey/Documents/Projet_contrastive/tensor_l_colors.pt")
# cmap = colors.Colormap(l_colors)
# print(cmap)
# print(liste[0])
# print("l",len(liste))
# print(len(l_colorstyh))
lights = torch.tensor(lights)
print(lights.shape)

# print(sdkjfdkjhf)

matrix2 = [] #should be shape = (56*100,128)
for i in range(len(liste)):
    # matrix2.append(torch.load(f"/CMF/data/timtey/results_contrastive_learning/results2/{liste[i]}"))
    # matrix2.append(torch.load(f"/CMF/data/timtey/results_contrastive_learning/results2_pretrained_true_t_04/{liste[i]}"))
    # aaa = torch.load(f"/CMF/data/timtey/results_contrastive_loss_combine/{liste[i]}")
    # print(aaa.shape)
    # print(aaa[0][-1], aaa[1][-1]) # data_lab
    # print(aaa[0][-2], aaa[1][-2]) # number cell
    # print(aaa[0][-3], aaa[1][-3]) # label tractography
    # print(aaa[0][-4], aaa[1][-4]) # number id
    # print(aaa[0][-5], aaa[1][-5]) # label
    # matrix2.append(torch.load(f"/CMF/data/timtey/results_contrastive_loss_combine/{liste[i]}"))
    matrix2.append(torch.load(f"/CMF/data/timtey/results_contrastive_loss_combine_loss_tract_cluster_with_simclr/{liste[i]}"))
    # matrix2 = torch.load("/CMF/data/timtey/results_contrastive_learning/results2/proj_testtensor([0], device='cuda:0')_612707.pt")
    # print("m",len(matrix2))
    # print("matrix2",type(matrix2))
    # matrix2 = torch.tensor(matrix2)
    # print(matrix2[0].shape)
    # matrix = matrix2[:,:128]
    # print(matrix.shape)
    # labels = matrix2[:,-1]
    # print(labels.shape)
    # print(labels)

MATR2 = torch.cat(matrix2, dim=0)
# print(MATR2.shape)
# print(MATR2[0])
# print(MATR2[1]) 
# print(MATR2[2])
Data_lab = MATR2[:,-1]
print(Data_lab.shape)
d_unique = torch.unique(Data_lab)
print(d_unique)
# MATR3
print(int(MATR2[0][-1]))
LAB = MATR2[:,-1]
print(LAB.shape)
MATR = MATR2[:,:128]
print(MATR.shape)
# print(sjdhfkjdg)

MATR_TRACT = torch.tensor([]).cuda()
MATR_BUNDLE = torch.tensor([]).cuda()
for i in range(MATR2.shape[0]):
    if MATR2[i][-1] == 1:
        MATR_TRACT = torch.cat((MATR_TRACT, MATR2[i].unsqueeze(0)), dim=0)
    elif MATR2[i][-1] == 2:
        MATR_BUNDLE = torch.cat((MATR_BUNDLE, MATR2[i].unsqueeze(0)), dim=0)

# print(MATR_TRACT.shape)
# print(MATR_BUNDLE.shape)
# print(MATR_TRACT[0][-1].item())
# print(MATR_BUNDLE[0][-1].item())
# print(MATR_TRACT[0][-2])
Data_lab_TRACT = MATR_TRACT[:,-1]
Data_lab_BUNDLE = MATR_BUNDLE[:,-1]
Number_cell_TRACT = MATR_TRACT[:,-2]
Number_cell_BUNDLE = MATR_BUNDLE[:,-2]
Label_tract_TRACT = MATR_TRACT[:,-3]
Label_tract_BUNDLE = MATR_BUNDLE[:,-3]
Number_id_TRACT = MATR_TRACT[:,-4]
Number_id_BUNDLE = MATR_BUNDLE[:,-4]
Label_TRACT = MATR_TRACT[:,-5]
Label_BUNDLE = MATR_BUNDLE[:,-5]
MATR_TRACT = MATR_TRACT[:,:128]
MATR_BUNDLE = MATR_BUNDLE[:,:128]
MATR_TRACT = MATR_TRACT.cpu()
MATR_BUNDLE = MATR_BUNDLE.cpu()
Data_lab_TRACT = Data_lab_TRACT.cpu()
Data_lab_BUNDLE = Data_lab_BUNDLE.cpu()
Number_cell_TRACT = Number_cell_TRACT.cpu()
Number_cell_BUNDLE = Number_cell_BUNDLE.cpu()
Label_tract_TRACT = Label_tract_TRACT.cpu()
Label_tract_BUNDLE = Label_tract_BUNDLE.cpu()
Number_id_TRACT = Number_id_TRACT.cpu()
Number_id_BUNDLE = Number_id_BUNDLE.cpu()
Label_TRACT = Label_TRACT.cpu()
Label_BUNDLE = Label_BUNDLE.cpu()
# print(MATR_TRACT.shape)
# print(MATR_BUNDLE.shape)
# print(Data_lab_TRACT.shape)
# print(Data_lab_BUNDLE.shape)
# print(Number_cell_TRACT.shape)
# print(Number_cell_BUNDLE.shape)
# print(Label_tract_TRACT.shape)
# print(Label_tract_BUNDLE.shape)
# print(Number_id_TRACT.shape)
# print(Number_id_BUNDLE.shape)
# print(Label_TRACT.shape)
# print(Label_BUNDLE.shape)
# print(skhgd)
# print(jhg)
# sum = 0
# for i in range(len(matrix)):
    # print(matrix[i].shape[0])
    # sum += matrix[i].shape[0]
# print(sum) # 11685 
# print(kajsghf)
print(MATR_BUNDLE.shape)
print(MATR_TRACT.shape)
# for i in range(MATR_BUNDLE.shape[0]):
#     norm =0
#     for j in range(MATR_BUNDLE.shape[1]):
#         norm += MATR_BUNDLE[i][j]**2
#     norm = norm**0.5
#     # print(i, norm)
#     MATR_BUNDLE[i] = MATR_BUNDLE[i]/norm
#     # print(i, norm)
# # print(dskjhgh)
# for a in range(MATR_BUNDLE.shape[0]):
#     norm =0
#     for b in range(MATR_BUNDLE.shape[1]):
#         norm += MATR_BUNDLE[a][b]**2
#     norm = norm**0.5
    # print(a, norm)
# print(skjfhkfdjh)
# MATR_TRACT = normalize(MATR_TRACT)
MATR_BUNDLE = MATR_BUNDLE/np.linalg.norm(MATR_BUNDLE, axis=1, keepdims=True)
MATR_TRACT = MATR_TRACT/np.linalg.norm(MATR_TRACT, axis=1, keepdims=True)



MATR = MATR.cpu()
LAB = LAB.cpu()
LIGHTS = lights.cpu()
# for i in range(6):
#     print(np.linalg.norm(MATR_BUNDLE[i]))
# print(np.linalg.norm(LIGHTS[0]))
# # print(MATR_BUNDLE[0], MATR_BUNDLE[1], MATR_BUNDLE[2], MATR_BUNDLE[3], MATR_BUNDLE[4], MATR_BUNDLE[5], MATR_BUNDLE[6])
# for i in range(6):
#     print(MATR_BUNDLE[i], Label_BUNDLE[i])
# print(LIGHTS[0])
# print(Label_BUNDLE)
# print(skjfkjsdh)
# print(LIGHTS[0], LIGHTS[0].shape)
# print(MATR_BUNDLE[0], MATR_BUNDLE[0].shape)
# print(MATR_BUNDLE[1])
# print(MATR_BUNDLE[2])
# print(MATR_BUNDLE[3])
# print(MATR_BUNDLE[4])
# print(sakfjfhhkdf)
# tsne = TSNE(n_components=2, random_state=10)
# print("l",LIGHTS[0])
# for i in range(50):
#     print(f"{i}",MATR_BUNDLE[i])

# pca = PCA(n_components=50)

threedtsne = TSNE(n_components=3, perplexity=500)
tsne = TSNE(n_components=2, perplexity=500)
threepca = PCA(n_components=3)
pca = PCA(n_components=2)
# Umap = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')
# tsne_results_test = tsne.fit_transform(MATR) #be careful matrix should be on cpu for this transformation
# tsne_results_test_test = tsne.fit_transform(MATR_TRACT, MATR_BUNDLE, LIGHTS) #be careful matrix should be on cpu for this transformation
MATR_TRACT = torch.tensor(MATR_TRACT)
MATR_BUNDLE = torch.tensor(MATR_BUNDLE)
MATR_TEST = torch.cat((MATR_TRACT, MATR_BUNDLE), dim=0)
print(MATR_TEST.shape)
# res_pca = pca.fit_transform(MATR_TEST)
# print("res_pca",res_pca.shape)

for i in range(205):
    for j in range(10):
        print(MATR_BUNDLE[i][0], MATR_BUNDLE[i][j],)
print(LIGHTS[0])

tsne_results_test_test = tsne.fit_transform(MATR_TEST) #be careful matrix should be on cpu for this transformation
threedtsne_results_test_test = threedtsne.fit_transform(MATR_TEST) #be careful matrix should be on cpu for this transformation
# tsne_results_test_test = pca.fit_transform(MATR_TEST) #be careful matrix should be on cpu for this transformation
# threedtsne_results_test_test = threepca.fit_transform(MATR_TEST) #be careful matrix should be on cpu for this transformation
# tsne_results_test_test = tsne.fit_transform(res_pca) #be careful matrix should be on cpu for this transformation
# threedtsne_results_test_test = threedtsne.fit_transform(res_pca) #be careful matrix should be on cpu for this transformation
# tsne_results_test_tract = tsne.fit_transform(MATR_TRACT) #be careful matrix should be on cpu for this transformation
# tsne_results_test_bundle = tsne.fit_transform(MATR_BUNDLE) #be careful matrix should be on cpu for this transformation
tsne_results_lights = tsne.fit_transform(LIGHTS) #be careful matrix should be on cpu for this transformation
threedtsne_results_lights = threedtsne.fit_transform(LIGHTS) #be careful matrix should be on cpu for this transformation
# tsne_results_lights = pca.fit_transform(LIGHTS) #be careful matrix should be on cpu for this transformation
# threedtsne_results_lights = threepca.fit_transform(LIGHTS) #be careful matrix should be on cpu for this transformation


# umap_results_test = Umap.fit_transform(MATR)
# plt.scatter(tsne_results_test[:, 0], tsne_results_test[:, 1], cmap='viridis')
# plt.colorbar()
# plt.show()
# torch.save(tsne_results_test_test, 'tsne_results_test_test.pt')
# torch.save(threedtsne_results_test_test, 'threedtsne_results_test_test.pt')
# torch.save(tsne_results_lights, 'tsne_results_lights.pt')
# torch.save(threedtsne_results_lights, 'threedtsne_results_lights.pt')

# tsne_results_test_test = torch.load('tsne_results_test_test.pt')
# tsne_results_lights = torch.load('tsne_results_lights.pt')
# threedtsne_results_test_test = torch.load('threedtsne_results_test_test.pt')
# threedtsne_results_lights = torch.load('threedtsne_results_lights.pt')

threedtsne_results_test_tract = threedtsne_results_test_test[:MATR_TRACT.shape[0],:]
# print(threedtsne_results_test_tract.shape)
threedtsne_results_test_bundle = threedtsne_results_test_test[MATR_TRACT.shape[0]:,:]
tsne_results_test_tract = tsne_results_test_test[:MATR_TRACT.shape[0],:]
# print(tsne_results_test_tract.shape)
tsne_results_test_bundle = tsne_results_test_test[MATR_TRACT.shape[0]:,:]
# print(tsne_results_test_bundle.shape)
# tsne_results_lights = tsne_results_test_test[MATR_TRACT.shape[0]+MATR_BUNDLE.shape[0]:,:]
# print(tsne_results_lights.shape)
# print(tsne_results_test_test.shape)
# print(tsne_results_lights)
# print(threedtsne_results_test_bundle.shape)
# print(threedtsne_results_test_tract.shape)
# for i in range(threedtsne_results_test_bundle.shape[0]):
    # print(f"{i}",np.linalg.norm(threedtsne_results_test_bundle[i]), np.linalg.norm(threedtsne_results_test_tract[i]))

# threedtsne_results_lights = threedtsne_results_lights/np.linalg.norm(threedtsne_results_lights, axis=1, keepdims=True)
# threedtsne_results_test_tract = threedtsne_results_test_tract/np.linalg.norm(threedtsne_results_test_tract, axis=1, keepdims=True)
# threedtsne_results_test_bundle = threedtsne_results_test_bundle/np.linalg.norm(threedtsne_results_test_bundle, axis=1, keepdims=True)
# tsne_results_lights = tsne_results_lights/np.linalg.norm(tsne_results_lights, axis=1, keepdims=True)
# tsne_results_test_tract = tsne_results_test_tract/np.linalg.norm(tsne_results_test_tract, axis=1, keepdims=True)
# tsne_results_test_bundle = tsne_results_test_bundle/np.linalg.norm(tsne_results_test_bundle, axis=1, keepdims=True)


threedtsne_results_lights_norm = np.abs(threedtsne_results_lights/np.linalg.norm(threedtsne_results_lights, axis=1, keepdims=True))
threedtsne_results_test_tract_norm = np.abs(threedtsne_results_test_tract/np.linalg.norm(threedtsne_results_test_tract, axis=1, keepdims=True))
threedtsne_results_test_bundle_norm = np.abs(threedtsne_results_test_bundle/np.linalg.norm(threedtsne_results_test_bundle, axis=1, keepdims=True))
tsne_results_lights_norm = np.abs(tsne_results_lights/np.linalg.norm(tsne_results_lights, axis=1, keepdims=True))
tsne_results_test_tract_norm = np.abs(tsne_results_test_tract/np.linalg.norm(tsne_results_test_tract, axis=1, keepdims=True))
tsne_results_test_bundle_norm = np.abs(tsne_results_test_bundle/np.linalg.norm(tsne_results_test_bundle, axis=1, keepdims=True))
# for i in range(threedtsne_results_test_bundle.shape[0]):
#     norm = 0
#     norm2 = 0
#     for j in range(threedtsne_results_test_bundle.shape[1]):
#         norm += threedtsne_results_test_bundle[i][j]**2
#         norm2 += threedtsne_results_test_tract[i][j]**2
#     norm = norm**0.5
#     norm2 = norm2**0.5
#     print(i, norm, norm2)



# # print(kdjfhdkjh)
# for i in range(threedtsne_results_lights.shape[0]):
#     norm = 0
#     for j in range(threedtsne_results_lights.shape[1]):
#         norm += threedtsne_results_lights[i][j]**2
#     norm = norm**0.5
#     print(i, norm)
# print(kdjfhdkjh)
# print("dsjhfgdjh")
# print(tsne_results_lights[0])
# print(tsne_results_test_bundle[0])
# # print(tsne_results_test_test[1], tsne_results_test_test[2], tsne_results_test_test[3], tsne_results_test_test[4], tsne_results_test_test[5])
# # print(tsne_results_test_test[6], tsne_results_test_test[7], tsne_results_test_test[8], tsne_results_test_test[9], tsne_results_test_test[10])
# mean_0 = []
# mean_1 = []
# for i in range(205):
#     print(f"{i}",tsne_results_test_bundle[i])
#     mean_0.append(tsne_results_test_bundle[i][0])
#     mean_1.append(tsne_results_test_bundle[i][1])
# print(mean_0, mean_1)
# print(np.mean(mean_0), np.mean(mean_1))
# print("fdjsgjhfgjhg")
kmean_lab = KMeans(n_clusters=57, random_state=0)
kmeans_labels = kmean_lab.fit_predict(tsne_results_test_bundle)
kmeans_labels_tract = kmean_lab.fit_predict(tsne_results_test_tract)
centers  = kmean_lab.cluster_centers_
centers_tract  = kmean_lab.cluster_centers_
# print("ceneters",centers)
# print(kmeans_labels.shape)
# print(type(kmeans_labels))
# print("kmeans_labels",kmeans_labels) # (11685,)
uniq = torch.unique(torch.tensor(kmeans_labels))
uniq = uniq.numpy()
uniq_tract = torch.unique(torch.tensor(kmeans_labels_tract))
uniq_tract = uniq_tract.numpy()

# uniq = [str(i) for i in uniq]
# print("unique",uniq)
# print(centers.shape)
# uniq_lab = torch.unique(LAB)
uniq_lab = torch.unique(Label_BUNDLE)
uniq_tract = torch.unique(Label_TRACT)
# for i in range(len(uniq_lab)):
# LAB = list(LAB)
Label_BUNDLE = list(Label_BUNDLE)
print("LAB",len(Label_BUNDLE))
Label_TRACT = list(Label_TRACT)
# print("LAB",len(LAB)) 
# print(tsne_results_test_tract.shape)
# print(tsne_results_test_bundle.shape)
# print("metrics", metrics.v_measure_score(LAB, kmeans_labels))

# 3d plot
# ax = plt.axes(projection='3d')
# ax.scatter(threedtsne_results_test_bundle[:, 0], threedtsne_results_test_bundle[:, 1], threedtsne_results_test_bundle[:, 2], c=Label_BUNDLE, s=50, cmap = l_colors11)
# for i in range(57):
#     ax = plt.axes(projection='3d')
#     ax.scatter(threedtsne_results_test_bundle[:(i+1)*205, 0], threedtsne_results_test_bundle[:(i+1)*205, 1], threedtsne_results_test_bundle[:(i+1)*250, 2], c=Label_BUNDLE[:(i+1)*250], s=50, cmap = l_colors11)
#     ax.scatter(threedtsne_results_lights[i, 0], threedtsne_results_lights[i, 1], threedtsne_results_lights[i, 2], s=150, cmap = l_colors11)
#     print("i",i)
#     plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# for i in range(len(Label_BUNDLE)):
    # if Label_BUNDLE[i] <20:
ax.scatter(threedtsne_results_test_bundle_norm[:4100:5, 0], threedtsne_results_test_bundle_norm[:4100:5, 1], threedtsne_results_test_bundle_norm[:4100:5, 2], c=Label_BUNDLE[:4100:5], s=50, cmap = l_colors1)
for i in range(20):
    ax.scatter(threedtsne_results_lights_norm[i, 0], threedtsne_results_lights_norm[i, 1], threedtsne_results_lights_norm[i, 2], linewidths=10)#, s=100, cmap = l_colors1)
# plt.colorbar(ax=ax)
plt.show()
ax = plt.axes(projection='3d')
# for i in range(len(Label_BUNDLE)):
#     if Label_BUNDLE[i]<40 and Label_BUNDLE[i]>=20:
ax.scatter(threedtsne_results_test_bundle_norm[4100:8200:5, 0], threedtsne_results_test_bundle_norm[4100:8200:5, 1], threedtsne_results_test_bundle_norm[4100:8200:5, 2], c=Label_BUNDLE[4100:8200:5], s=50, cmap = l_colors2)
for i in range(20,40):
    ax.scatter(threedtsne_results_lights_norm[i, 0], threedtsne_results_lights_norm[i, 1], threedtsne_results_lights_norm[i, 2], linewidths=10)#, s=100, cmap = l_colors2)
# plt.colorbar(ax=ax)
plt.show()
ax = plt.axes(projection='3d')
# for i in range(len(Label_BUNDLE)):
#     if Label_BUNDLE[i]>=40:
ax.scatter(threedtsne_results_test_bundle_norm[8200::5, 0], threedtsne_results_test_bundle_norm[8200::5, 1], threedtsne_results_test_bundle_norm[8200::5, 2], c=Label_BUNDLE[8200::5], s=50, cmap = l_colors3)
for i in range(40,57):
    ax.scatter(threedtsne_results_lights_norm[i, 0], threedtsne_results_lights_norm[i, 1], threedtsne_results_lights_norm[i, 2], linewidths=10)#, s=100, cmap = l_colors3)
# plt.colorbar(ax=ax)
plt.show()

ax = plt.axes(projection='3d')
ax.scatter(threedtsne_results_test_bundle_norm[::5, 0], threedtsne_results_test_bundle_norm[::5, 1], threedtsne_results_test_bundle_norm[::5, 2], c=Label_BUNDLE[::5], s=50, cmap=l_colors)
for i in range(threedtsne_results_lights_norm.shape[0]):
    ax.scatter(threedtsne_results_lights_norm[i, 0], threedtsne_results_lights_norm[i, 1], threedtsne_results_lights_norm[i, 2], linewidths=10)#, s=100, cmap=l_colors)# plt.colorbar(ticks = uniq_lab)
# ax.plt.colorbar(ticks = uniq_lab)
# plt.colorbar(ax=ax)
plt.show()

ax = plt.axes(projection='3d')
ax.scatter(threedtsne_results_test_tract_norm[:, 0], threedtsne_results_test_tract_norm[:, 1], threedtsne_results_test_tract_norm[:, 2], c=Label_TRACT, s=50, cmap=l_colors)
for i in range(threedtsne_results_lights_norm.shape[0]):
    ax.scatter(threedtsne_results_lights_norm[i, 0], threedtsne_results_lights_norm[i, 1], threedtsne_results_lights_norm[i, 2], s=100, cmap=l_colors)
# plt.colorbar(ax=ax)
plt.show()




print("metrics", metrics.v_measure_score(Label_BUNDLE, kmeans_labels))
plt.scatter(tsne_results_test_tract[:, 0], tsne_results_test_tract[:, 1],s=50)
plt.scatter(tsne_results_test_bundle[:, 0], tsne_results_test_bundle[:, 1], c=Label_BUNDLE,s=50, cmap=l_colors)
# for i in range(len(centers)):
    # plt.text(centers[i, 0], centers[i, 1], str(uniq[i]), fontsize=12)
for i in range(LIGHTS.shape[0]):
    plt.text(tsne_results_lights[i, 0], tsne_results_lights[i, 1], str(i), fontsize=12)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(tsne_results_test[:, 0], tsne_results_test[:, 1], tsne_results_test[:, 2], c=LAB,s=50, cmap=l_colors)
# for i in range(len(LAB)):
    # plt.text(tsne_results_test[i, 0], tsne_results_test[i, 1], str(LAB[i]), fontsize=12)
plt.colorbar(ticks = uniq_lab)
# plt.colorbar()
# plt.xlim(-100,100)
# plt.ylim(-100,100)
# plt.xlim(-50,50)
# plt.ylim(-50,50)
plt.show()
plt.scatter(tsne_results_test_bundle[:, 0], tsne_results_test_bundle[:, 1], c=Label_BUNDLE,s=50, cmap=l_colors)
for i in range(LIGHTS.shape[0]):
    plt.text(tsne_results_lights[i, 0], tsne_results_lights[i, 1], str(i), fontsize=12)
plt.colorbar(ticks = uniq_lab)
# plt.xlim(-100,100)
# plt.ylim(-100,100)
# plt.xlim(-50,50)
# plt.ylim(-50,50)
plt.show()

plt.scatter(tsne_results_test_tract[:, 0], tsne_results_test_tract[:, 1], c=Label_TRACT,s=50, cmap=l_colors)
plt.colorbar(ticks = uniq_tract)
# plt.xlim(-100,100)
# plt.ylim(-100,100)
# plt.xlim(-50,50)
# plt.ylim(-50,50)
plt.show()
# plt.subplot(1,3,1)
# print(sakjhfkjh)

plt.scatter(tsne_results_test_bundle[:4100, 0], tsne_results_test_bundle[:4100, 1], c=LAB[:4100],s=50, cmap=l_colors1)
plt.colorbar(ticks = uniq_lab)
# plt.xlim(-30,30)
# plt.ylim(-30,30)
# plt.xlim(-100,100)
# plt.ylim(-100,100)
# plt.xlim(-50,50)
# plt.ylim(-50,50)
plt.show()
# plt.subplot(1,3,2)
plt.scatter(tsne_results_test_bundle[4100:8200, 0], tsne_results_test_bundle[4100:8200, 1], c=LAB[4100:8200],s=50, cmap=l_colors2)
plt.colorbar(ticks = uniq_lab)
# plt.xlim(-30,30)
# plt.ylim(-30,30)
# plt.xlim(-100,100)
# plt.ylim(-100,100)
# plt.xlim(-50,50)
# plt.ylim(-50,50)
plt.show()
# plt.subplot(1,3,3)
# plt.scatter(tsne_results_test_bundle[8200:12300, 0], tsne_results_test_bundle[8200:12300, 1], c=LAB[8200:12300],s=50, cmap=l_colors3)
# plt.colorbar(ticks = uniq_lab)
# # plt.xlim(-30,30)
# # plt.ylim(-30,30)
# plt.xlim(-100,100)
# plt.ylim(-100,100)
# plt.show()



# print("jzhfdsg",kmeans_labels.shape)
plt.scatter(tsne_results_test_bundle[:, 0], tsne_results_test_bundle[:, 1], c=kmeans_labels,s=50, cmap='viridis')
print(str(uniq[0]))
for i in range(len(centers)):
    plt.text(centers[i, 0], centers[i, 1], str(uniq[i]), fontsize=12)
# plt.text(centers[:, 0], centers[:, 1], 'coucou', fontsize=12)
# plt.text(3.5, 0.9, 'Sine wave', fontsize = 23)
# kmeans_labels = kmeans_labels.to("cuda:0")
# kmeans_labels = torch.tensor(kmeans_labels)
# centers = kmeans_labels.cluster_centers_
# print(centers.shape)
# print(centers)
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.contourf(tsne_results_test[:, 0], tsne_results_test[:, 1], kmeans_labels, cmap='viridis')
plt.colorbar()
# plt.legend()
# plt.xlim(-100,100)
# plt.ylim(-100,100)
# plt.xlim(-50,50)
# plt.ylim(-50,50)
plt.show()
# print(str(uniq[0]))
# for i in range(len(centers)):
# plt.text(centers[:, 0], centers[:, 1], "coucou", fontsize=12)
# plt.colorbar()
# plt.show()
# print(tsne_results_test)
# print(kmeans_labels)
# filtered_0 = tsne_results_test[kmeans_labels==0]
# print(filtered_0.shape)
# print(filtered_0)
# plt.scatter(filtered_0[:, 0], filtered_0[:, 1],kmeans_labels[0], cmap='viridis')
# plt.colorbar()
# plt.show()
# for i in uniq:
    # plt.scatter(tsne_results_test[kmeans_labels==i, 0], tsne_results_test[kmeans_labels==i, 1])
# plt.legend()
# plt.show()
# print(MATR2.shape)
# print(len(MATR2[0]))
# print(kmeans_labels.shape)
# print(kmean_lab)

# for i in range(len(uniq)):
    # if int(LAB[i]) == i:
        # 

for i in range(len(uniq)):
    print("i",i)
    l = []
    l_0 = []
    l_1 = []
    km = []
    c_l = []
    # print("MATR_BUNDLE.shape[0]",Label_BUNDLE.shape)
    for j in range(len(Label_BUNDLE)):
        # print(Label_BUNDLE[j])
        if int(Label_BUNDLE[j]) == i :
            l.append(tsne_results_test_bundle[j])
            # print("j", tsne_results_test[j,0])
            l_0.append(tsne_results_test_bundle[j,0])
            l_1.append(tsne_results_test_bundle[j,1])
            # plt.scatter(tsne_results_test[j, 0], tsne_results_test[j, 1], c =i ,s=50, alpha=0.2, cmap='viridis')
            km.append(kmeans_labels[j])
            # c_l.append(kmeans_labels[j])
    # print("l",len(l))
    # print(l[0,0])
    # print(l[0,1])
    # print(l[0])
    lc_map = colors.ListedColormap(np.random.rand(len(torch.unique(torch.tensor(km))),3))
    plt.scatter(l_0, l_1, c =km[:] ,s=50, alpha=0.5, cmap=lc_map)
    # plt.xlim(-100,100)
    # plt.ylim(-100,100)
    # plt.xlim(-50,50)
    # plt.ylim(-50,50)
    plt.colorbar(ticks = torch.unique(torch.tensor(km)))
    plt.title("cluster "+str(i))
        # plt.colorbar()
    nb = []
    for ele in range(len(l)):
        bn = []
        for ele2 in range(len(l)):
            eq = (l[ele][0]-l[ele2][0])**2 + (l[ele][1]-l[ele2][1])**2
            if eq < 20:
                bn.append([ele2,l[ele][0], l[ele][1]])
        nb.append([len(bn),l[ele][0], l[ele][1]])
    # print("nb",nb)
    # print("nb",max(nb[0]))
    nb2 = sorted(nb)
    for i in range(len(nb2)):
        if nb2[i][0]>15:
            print(nb2[i])
    # maaxi = nb.index(max(nb))
    # print("maaxi",nb[maaxi])
    # for n in range(len(nb)):
    #     if nb[n][0]>(max(nb[0])-3):
    #         print(nb[n])
    # nb3 = nb
    # nb3[:][0] = -1
    # print("nb3",nb3)
    # for i in range(len(nb3)):
    #     if nb3[i]
    print("km",torch.tensor(km).unique(return_counts=True))
    print("km",km)
    l_true = [i]*len(km)
    print("metrics", metrics.v_measure_score(l_true, km))
    plt.show()
    # plt.savefig(f"/home/timtey/Documents/Projet_contrastive/results_lab/cluster{i}.png")

