import numpy as np
import torch

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os
from sklearn import metrics
import matplotlib.colors as colors
import umap
import pandas as pd

lights = pd.read_pickle(r'Lights_good.pickle')
liste = os.listdir("/CMF/data/timtey/results_contrastive_loss_combine_loss_tract_cluster_with_simclr")
np.random.seed(125467)
l_colors = colors.ListedColormap ( np.random.rand (57,3))
l_colors1 = colors.ListedColormap ( np.random.rand (20,3))
l_colors2 = colors.ListedColormap ( np.random.rand (20,3))
l_colors3 = colors.ListedColormap ( np.random.rand (17,3))
l_colors11 = colors.ListedColormap ( np.random.rand (1,3))
lights = torch.tensor(lights)

matrix2 = [] #should be shape = (56*100,128)
for i in range(len(liste)):
    matrix2.append(torch.load(f"/CMF/data/timtey/results_contrastive_loss_combine_loss_tract_cluster_with_simclr/{liste[i]}"))

MATR2 = torch.cat(matrix2, dim=0)
Data_lab = MATR2[:,-1]
d_unique = torch.unique(Data_lab)
LAB = MATR2[:,-1]
MATR = MATR2[:,:128]

MATR_TRACT = torch.tensor([]).cuda()
MATR_BUNDLE = torch.tensor([]).cuda()
for i in range(MATR2.shape[0]):
    if MATR2[i][-1] == 1:
        MATR_TRACT = torch.cat((MATR_TRACT, MATR2[i].unsqueeze(0)), dim=0)
    elif MATR2[i][-1] == 2:
        MATR_BUNDLE = torch.cat((MATR_BUNDLE, MATR2[i].unsqueeze(0)), dim=0)

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

MATR_BUNDLE = MATR_BUNDLE/np.linalg.norm(MATR_BUNDLE, axis=1, keepdims=True)
MATR_TRACT = MATR_TRACT/np.linalg.norm(MATR_TRACT, axis=1, keepdims=True)

MATR = MATR.cpu()
LAB = LAB.cpu()
LIGHTS = lights.cpu()

threedtsne = TSNE(n_components=3, perplexity=500)
tsne = TSNE(n_components=2, perplexity=500)
threepca = PCA(n_components=3)
pca = PCA(n_components=2)
MATR_TRACT = torch.tensor(MATR_TRACT)
MATR_BUNDLE = torch.tensor(MATR_BUNDLE)
MATR_TEST = torch.cat((MATR_TRACT, MATR_BUNDLE), dim=0)

for i in range(205):
    for j in range(10):
        print(MATR_BUNDLE[i][0], MATR_BUNDLE[i][j],)

tsne_results_test_test = tsne.fit_transform(MATR_TEST) #be careful matrix should be on cpu for this transformation
threedtsne_results_test_test = threedtsne.fit_transform(MATR_TEST) #be careful matrix should be on cpu for this transformation
tsne_results_lights = tsne.fit_transform(LIGHTS) #be careful matrix should be on cpu for this transformation
threedtsne_results_lights = threedtsne.fit_transform(LIGHTS) #be careful matrix should be on cpu for this transformation

threedtsne_results_test_tract = threedtsne_results_test_test[:MATR_TRACT.shape[0],:]
threedtsne_results_test_bundle = threedtsne_results_test_test[MATR_TRACT.shape[0]:,:]
tsne_results_test_tract = tsne_results_test_test[:MATR_TRACT.shape[0],:]
tsne_results_test_bundle = tsne_results_test_test[MATR_TRACT.shape[0]:,:]

threedtsne_results_lights_norm = np.abs(threedtsne_results_lights/np.linalg.norm(threedtsne_results_lights, axis=1, keepdims=True))
threedtsne_results_test_tract_norm = np.abs(threedtsne_results_test_tract/np.linalg.norm(threedtsne_results_test_tract, axis=1, keepdims=True))
threedtsne_results_test_bundle_norm = np.abs(threedtsne_results_test_bundle/np.linalg.norm(threedtsne_results_test_bundle, axis=1, keepdims=True))
tsne_results_lights_norm = np.abs(tsne_results_lights/np.linalg.norm(tsne_results_lights, axis=1, keepdims=True))
tsne_results_test_tract_norm = np.abs(tsne_results_test_tract/np.linalg.norm(tsne_results_test_tract, axis=1, keepdims=True))
tsne_results_test_bundle_norm = np.abs(tsne_results_test_bundle/np.linalg.norm(tsne_results_test_bundle, axis=1, keepdims=True))

kmean_lab = KMeans(n_clusters=57, random_state=0)
kmeans_labels = kmean_lab.fit_predict(tsne_results_test_bundle)
kmeans_labels_tract = kmean_lab.fit_predict(tsne_results_test_tract)
centers  = kmean_lab.cluster_centers_
centers_tract  = kmean_lab.cluster_centers_
uniq = torch.unique(torch.tensor(kmeans_labels))
uniq = uniq.numpy()
uniq_tract = torch.unique(torch.tensor(kmeans_labels_tract))
uniq_tract = uniq_tract.numpy()

uniq_lab = torch.unique(Label_BUNDLE)
uniq_tract = torch.unique(Label_TRACT)
Label_BUNDLE = list(Label_BUNDLE)
Label_TRACT = list(Label_TRACT)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(threedtsne_results_test_bundle_norm[:4100:5, 0], threedtsne_results_test_bundle_norm[:4100:5, 1], threedtsne_results_test_bundle_norm[:4100:5, 2], c=Label_BUNDLE[:4100:5], s=50, cmap = l_colors1)
for i in range(20):
    ax.scatter(threedtsne_results_lights_norm[i, 0], threedtsne_results_lights_norm[i, 1], threedtsne_results_lights_norm[i, 2], linewidths=10)#, s=100, cmap = l_colors1)
# plt.colorbar(ax=ax)
plt.show()
ax = plt.axes(projection='3d')
ax.scatter(threedtsne_results_test_bundle_norm[4100:8200:5, 0], threedtsne_results_test_bundle_norm[4100:8200:5, 1], threedtsne_results_test_bundle_norm[4100:8200:5, 2], c=Label_BUNDLE[4100:8200:5], s=50, cmap = l_colors2)
for i in range(20,40):
    ax.scatter(threedtsne_results_lights_norm[i, 0], threedtsne_results_lights_norm[i, 1], threedtsne_results_lights_norm[i, 2], linewidths=10)#, s=100, cmap = l_colors2)
# plt.colorbar(ax=ax)
plt.show()
ax = plt.axes(projection='3d')
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
for i in range(LIGHTS.shape[0]):
    plt.text(tsne_results_lights[i, 0], tsne_results_lights[i, 1], str(i), fontsize=12)
plt.colorbar(ticks = uniq_lab)
plt.show()
plt.scatter(tsne_results_test_bundle[:, 0], tsne_results_test_bundle[:, 1], c=Label_BUNDLE,s=50, cmap=l_colors)
for i in range(LIGHTS.shape[0]):
    plt.text(tsne_results_lights[i, 0], tsne_results_lights[i, 1], str(i), fontsize=12)
plt.colorbar(ticks = uniq_lab)
plt.show()

plt.scatter(tsne_results_test_tract[:, 0], tsne_results_test_tract[:, 1], c=Label_TRACT,s=50, cmap=l_colors)
plt.colorbar(ticks = uniq_tract)
plt.show()

plt.scatter(tsne_results_test_bundle[:4100, 0], tsne_results_test_bundle[:4100, 1], c=LAB[:4100],s=50, cmap=l_colors1)
plt.colorbar(ticks = uniq_lab)
plt.show()

plt.scatter(tsne_results_test_bundle[4100:8200, 0], tsne_results_test_bundle[4100:8200, 1], c=LAB[4100:8200],s=50, cmap=l_colors2)
plt.colorbar(ticks = uniq_lab)
plt.show()

plt.scatter(tsne_results_test_bundle[:, 0], tsne_results_test_bundle[:, 1], c=kmeans_labels,s=50, cmap='viridis')
print(str(uniq[0]))
for i in range(len(centers)):
    plt.text(centers[i, 0], centers[i, 1], str(uniq[i]), fontsize=12)
plt.colorbar()
plt.show()

for i in range(len(uniq)):
    print("i",i)
    l = []
    l_0 = []
    l_1 = []
    km = []
    c_l = []
    for j in range(len(Label_BUNDLE)):
        if int(Label_BUNDLE[j]) == i :
            l.append(tsne_results_test_bundle[j])
            l_0.append(tsne_results_test_bundle[j,0])
            l_1.append(tsne_results_test_bundle[j,1])
            km.append(kmeans_labels[j])
    lc_map = colors.ListedColormap(np.random.rand(len(torch.unique(torch.tensor(km))),3))
    plt.scatter(l_0, l_1, c =km[:] ,s=50, alpha=0.5, cmap=lc_map)
    plt.colorbar(ticks = torch.unique(torch.tensor(km)))
    plt.title("cluster "+str(i))
    nb = []
    for ele in range(len(l)):
        bn = []
        for ele2 in range(len(l)):
            eq = (l[ele][0]-l[ele2][0])**2 + (l[ele][1]-l[ele2][1])**2
            if eq < 20:
                bn.append([ele2,l[ele][0], l[ele][1]])
        nb.append([len(bn),l[ele][0], l[ele][1]])
    nb2 = sorted(nb)
    for i in range(len(nb2)):
        if nb2[i][0]>15:
            print(nb2[i])
    print("km",torch.tensor(km).unique(return_counts=True))
    print("km",km)
    l_true = [i]*len(km)
    print("metrics", metrics.v_measure_score(l_true, km))
    plt.show()

