import numpy as np
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl 
import torchvision.models as models
from torch.nn.functional import softmax
import torchmetrics
import utils
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

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
# import MLP
import random
import os
from sklearn import metrics
# import random
import matplotlib.colors as colors
import umap
# get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
# l_colors = get_colors(57) # sample return:  ['#8af5da', '#fbc08c', '#b741d0']
# l_colors = plt.colors.ListedColormap ( np.random.rand ( 256,57))
# l_colors = colorbar.Colormap()
# liste = os.listdir("/CMF/data/timtey/results_contrastive_learning/results2")
liste = os.listdir("/CMF/data/timtey/results_contrastive_learning/results2_pretrained_true_t_04")
# l_colors = ['#DF2F0A', '#F85904', '#FFA500', '#FFD700', '#FFFF00', '#ADFF2F', '#7FFF00', '#00FF00', '#00FA9A', '#00FFFF', '#00BFFF', '#1E90FF', '#0000FF', '#8A2BE2', '#FF00FF', '#FF1493', '#FFC0CB', '#FFA07A', '#FF4500', '#A52A2A', '#808080', '#000000', '#FFFFFF', '#FFDAB9', '#FFE4B5', '#F0E68C', '#EEE8AA', '#BDB76B', '#556B2F', '#006400', '#008000', '#228B22', '#2E8B57', '#3CB371', '#20B2AA', '#00FFFF', '#00CED1', '#1E90FF', '#0000FF', '#000080', '#483D8B', '#8A2BE2', '#9932CC', '#FF00FF', '#FF1493', '#C71585', '#FFC0CB', '#FFA07A', '#FF4500', '#A52A2A', '#808080', '#000000', '#FFFFFF', '#FFDAB9', '#FFE4B5', '#F0E68C', '#EEE8AA', '#BDB76B', '#556B2F', '#006400', '#008000', '#228B22', '#2E8B57', '#3CB371', '#20B2AA', '#00FFFF', '#00CED1', '#1E90FF', '#0000FF', '#000080', '#483D8B', '#8A2BE2', '#9932CC', '#FF00FF', '#FF1493', '#C71585', '#FFC0CB', '#FFA07A', '#FF4500', '#A52A2A', '#808080', '#000000', '#FFFFFF', '#FFDAB9', '#FFE4B5', '#F0E68C', '#EEE8AA', '#BDB76B', '#556B2F', '#006400', '#008000', '#228B22', '#2E8B57', '#3CB371', '#20B2AA', '#00FFFF', '#00CED1', '#1E90FF', '#0000FF', '#000080', '#483D8B', '#8A2BE2', '#9932CC', '#FF00FF', '#FF1493', '#C71585', '#FFC0CB', '#FFA07A', '#FF4500', '#A52A2A', '#808080', '#000000', '#FFFFFF', '#FFDAB9', '#FFE4B5', '#F0E68C', '#EEE8AA']
l_colors = colors.ListedColormap ( np.random.rand (57,3))
l_colors1 = colors.ListedColormap ( np.random.rand (20,3))
l_colors2 = colors.ListedColormap ( np.random.rand (20,3))
l_colors3 = colors.ListedColormap ( np.random.rand (17,3))
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
matrix2 = [] #should be shape = (56*100,128)
for i in range(len(liste)):
    # matrix2.append(torch.load(f"/CMF/data/timtey/results_contrastive_learning/results2/{liste[i]}"))
    matrix2.append(torch.load(f"/CMF/data/timtey/results_contrastive_learning/results2_pretrained_true_t_04/{liste[i]}"))
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
print(int(MATR2[0][-1]))
LAB = MATR2[:,-1]
print(LAB)
MATR = MATR2[:,:128]
print(MATR.shape)
# print(sjdhfkjdg)
# print(skhgd)
# print(jhg)
# sum = 0
# for i in range(len(matrix)):
    # print(matrix[i].shape[0])
    # sum += matrix[i].shape[0]
# print(sum) # 11685 
# print(kajsghf)
MATR = MATR.cpu()
LAB = LAB.cpu()
# tsne = TSNE(n_components=2, random_state=10)
tsne = TSNE(n_components=2, perplexity=205)
# Umap = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')
tsne_results_test = tsne.fit_transform(MATR) #be careful matrix should be on cpu for this transformation
# umap_results_test = Umap.fit_transform(MATR)
# plt.scatter(tsne_results_test[:, 0], tsne_results_test[:, 1], cmap='viridis')
# plt.colorbar()
# plt.show()

kmean_lab = KMeans(n_clusters=57, random_state=0)
kmeans_labels = kmean_lab.fit_predict(tsne_results_test)
centers  = kmean_lab.cluster_centers_
# print("ceneters",centers)
# print(kmeans_labels.shape)
# print(type(kmeans_labels))
# print("kmeans_labels",kmeans_labels) # (11685,)
uniq = torch.unique(torch.tensor(kmeans_labels))
uniq = uniq.numpy()
# uniq = [str(i) for i in uniq]
# print("unique",uniq)
# print(centers.shape)
uniq_lab = torch.unique(LAB)
# for i in range(len(uniq_lab)):
LAB = list(LAB)
# print("LAB",len(LAB)) 
print("metrics", metrics.v_measure_score(LAB, kmeans_labels))
plt.scatter(tsne_results_test[:, 0], tsne_results_test[:, 1], c=LAB,s=50, cmap=l_colors)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(tsne_results_test[:, 0], tsne_results_test[:, 1], tsne_results_test[:, 2], c=LAB,s=50, cmap=l_colors)
# for i in range(len(LAB)):
    # plt.text(tsne_results_test[i, 0], tsne_results_test[i, 1], str(LAB[i]), fontsize=12)
plt.colorbar(ticks = uniq_lab)
# plt.colorbar()
# plt.xlim(-30,30)
# plt.ylim(-30,30)
plt.show()
# plt.subplot(1,3,1)


plt.scatter(tsne_results_test[:4100, 0], tsne_results_test[:4100, 1], c=LAB[:4100],s=50, cmap=l_colors1)
plt.colorbar(ticks = uniq_lab)
# plt.xlim(-30,30)
# plt.ylim(-30,30)
plt.xlim(-100,100)
plt.ylim(-100,100)
plt.show()
# plt.subplot(1,3,2)
plt.scatter(tsne_results_test[4100:8200, 0], tsne_results_test[4100:8200, 1], c=LAB[4100:8200],s=50, cmap=l_colors2)
plt.colorbar(ticks = uniq_lab)
# plt.xlim(-30,30)
# plt.ylim(-30,30)
plt.xlim(-100,100)
plt.ylim(-100,100)
plt.show()
# plt.subplot(1,3,3)
plt.scatter(tsne_results_test[8200:, 0], tsne_results_test[8200:, 1], c=LAB[8200:],s=50, cmap=l_colors3)
plt.colorbar(ticks = uniq_lab)
# plt.xlim(-30,30)
# plt.ylim(-30,30)
plt.xlim(-100,100)
plt.ylim(-100,100)
plt.show()



# print("jzhfdsg",kmeans_labels.shape)
plt.scatter(tsne_results_test[:, 0], tsne_results_test[:, 1], c=kmeans_labels,s=50, cmap='viridis')
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
print(MATR2.shape)
print(len(MATR2[0]))
print(kmeans_labels.shape)
print(kmean_lab)

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
    for j in range(MATR2.shape[0]):
        if int(MATR2[j][-1]) == i :
            l.append(tsne_results_test[j])
            # print("j", tsne_results_test[j,0])
            l_0.append(tsne_results_test[j,0])
            l_1.append(tsne_results_test[j,1])
            # plt.scatter(tsne_results_test[j, 0], tsne_results_test[j, 1], c =i ,s=50, alpha=0.2, cmap='viridis')
            km.append(kmeans_labels[j])
            # c_l.append(kmeans_labels[j])
    # print("l",len(l))
    # print(l[0,0])
    # print(l[0,1])
    # print(l[0])
    lc_map = colors.ListedColormap(np.random.rand(len(torch.unique(torch.tensor(km))),3))
    plt.scatter(l_0, l_1, c =km[:] ,s=50, alpha=0.5, cmap=lc_map)
    plt.xlim(-120,120)
    plt.ylim(-120,120)
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

