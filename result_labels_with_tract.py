import numpy as np
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import random
import os
import matplotlib.colors as colors
import umap
import pandas as pd
import math as m

lights = pd.read_pickle(r'lights_57_128d_on_sphere.pickle')
liste = os.listdir("/CMF/data/timtey/results_contrastive_learning_063023")
l_colors = colors.ListedColormap ( np.random.rand (57,3))
l_colors1 = colors.ListedColormap ( np.random.rand (20,3))
l_colors2 = colors.ListedColormap ( np.random.rand (20,3))
l_colors3 = colors.ListedColormap ( np.random.rand (17,3))

lights = torch.tensor(lights)
matrix2 = [] #should be shape = (56*100,128)
matrix_1 = []
matrix_2 = []
for i in range(len(liste)):
    matrix = torch.load(f"/CMF/data/timtey/results_contrastive_learning_063023/{liste[i]}")
    matrix_bundle = matrix[:matrix.shape[0]//2]
    matrix_tract = matrix[matrix.shape[0]//2:]
    matrix2.append(matrix)
    matrix_1.append(matrix_bundle)
    matrix_2.append(matrix_tract)
MATR2 = torch.cat(matrix2, dim=0)
MATR_1 = torch.cat(matrix_1, dim=0)
MATR_2 = torch.cat(matrix_2, dim=0)
Data_lab = MATR2[:,-1]
Data_lab_1 = MATR_1[:,-1]
Data_lab_2 = MATR_2[:,-1]
d_unique = torch.unique(Data_lab)

LAB = MATR2[:,-5]
LAB_1 = MATR_1[:,-5]
LAB_2 = MATR_2[:,-5]

n_lab = MATR_2[:,-4:-1]
MATR = MATR2[:,:128]
MATR = MATR.cpu()
LAB = LAB.cpu()
LIGHTS = lights.cpu()
MATRB = MATR_1[:,:128]
MATRB = MATRB.cpu()
MATRT = MATR_2[:,:128]
MATRT = MATRT.cpu()

LAB_1 = LAB_1.cpu()
LAB_2 = LAB_2.cpu()
n_lab = n_lab.cpu()
Mean_Pos = torch.tensor([])
for i in range(57):
    L_i = []
    for j in range(len(MATRB)):
        if LAB_1[j]==i:
            L_i.append(j)
    mean_pos =torch.zeros(128)
    MATRB_i = MATRB[L_i,:]
    for k in range(MATRB_i.shape[1]):
        mean_pos[k] = torch.mean(MATRB_i[:,k])
    mean_pos = mean_pos.unsqueeze(0)
    Mean_Pos = torch.cat((Mean_Pos, mean_pos), dim=0)

MATRT = MATRT.to('cuda:1')
MATRB = MATRB.to('cuda:1')
LIGHTS = LIGHTS.to('cuda:1')
L_D_128D = []
LAB_TRACT = []
LAB_tract_i = []
seuil128d = 0.3
L_D_128D_int = []
N_LAB = []
MeanD = torch.tensor([])
STDD = torch.tensor([])
for i in range(len(Mean_Pos)):
    L_i_128D = []
    L_n_lab_i = []
    Dist_i = torch.zeros(MATRT.shape[0], MATRT.shape[1])
    for j in range(MATRT.shape[1]):
        Dist_i[:,j] = (Mean_Pos[i,j]-MATRT[:,j])**2
    Dist_i = torch.sum(Dist_i, dim=1)
    Dist_i = torch.sqrt(Dist_i)
    for a in range(len(Dist_i)):
        if Dist_i[a]<seuil128d:
            L_i_128D.append(a)
    L_D_128D.extend(L_i_128D)
    MATRT_i = MATRT[L_i_128D,:]
    dist_i = torch.zeros(MATRT_i.shape[0], MATRT_i.shape[1])
    for j in range(MATRT_i.shape[1]):
        dist_i[:,j] = (Mean_Pos[i,j]-MATRT_i[:,j])**2
    dist_i = torch.sum(dist_i, dim=1)
    dist_i = torch.sqrt(dist_i)
    mean_dist = torch.mean(dist_i)
    std_dist = torch.std(dist_i)
    MeanD = torch.cat((MeanD, mean_dist.unsqueeze(0)), dim=0)
    STDD = torch.cat((STDD, std_dist.unsqueeze(0)), dim=0)
    for k in range(len(MATRT_i)):
        if dist_i[k]<mean_dist+std_dist:
            L_D_128D_int.append(L_i_128D[k])
            LAB_2[L_i_128D[k]] = i
            L_n_lab_i.append(n_lab[L_i_128D[k]])
    print("L_D_128D_int : ", len(L_D_128D_int))
    N_LAB.append(L_n_lab_i)

L_D_128D_int = sorted(L_D_128D_int)
print("sorted", L_D_128D_int)
L_D_128D_int = list(np.unique(L_D_128D_int))
print("unique", L_D_128D_int)
print("nb_closest points", len(L_D_128D_int))
print(len(L_D_128D_int))

print("N_LABELS", len(N_LAB))
L_reject = []
for i in range(MATRT.shape[0]):
    if i not in L_D_128D_int:
        L_reject.append(i)
print("L_reject", len(L_reject))

# L_D_128D = sorted(L_D_128D)
# L_D_128D = list(np.unique(L_D_128D))
# print("nb_closest points", len(L_D_128D))
# print(len(L_D_128D_int))

# file_name = "L_reject.pkl"
# open_file = open(file_name, "wb")
# pickle.dump(L_reject, open_file)
# open_file.close()
MATRT = MATRT.cpu()
MATRB = MATRB.cpu()
LIGHTS = LIGHTS.cpu()

MATRT = MATRT[L_D_128D_int,:]
LAB_2 = LAB_2[L_D_128D_int]
# Data_lab_2 = Data_lab_2[L_D_128D_int]
print("lab2", LAB_2.shape)
print("matrt", MATRT.shape)
print(LAB_2.shape)

to_reject = n_lab[L_reject]
print("to_reject", to_reject.shape)
# file_name = "to_reject.pkl"
# open_file = open(file_name, "wb")
# pickle.dump(to_reject, open_file)
# open_file.close()

n_lab = n_lab[L_D_128D_int]
print("n_lab", n_lab.shape)
to_save = torch.cat((n_lab, LAB_2.unsqueeze(1)), dim=1)


threedpca = PCA(n_components=3)
twopca = PCA(n_components=2)
'''
GLOBAL = torch.cat((MATRB,MATRT,LIGHTS,Mean_Pos),dim=0)
print(GLOBAL.shape)
threedpca_results = threedpca.fit_transform(GLOBAL)
twopca_results = twopca.fit_transform(GLOBAL)
print(threedpca_results.shape)
print(twopca_results.shape)
threedpca_bundle = threedpca_results[:MATRB.shape[0],:]
threedpca_tract = threedpca_results[MATRB.shape[0]:MATRB.shape[0]+MATRT.shape[0],:]
threedpca_lights = threedpca_results[MATRB.shape[0]+MATRT.shape[0]:MATRB.shape[0]+MATRT.shape[0]+Mean_Pos.shape[0],:]
threedpca_meanpos = threedpca_results[MATRB.shape[0]+MATRT.shape[0]+Mean_Pos.shape[0]:,:]
twopca_bundle = twopca_results[:MATRB.shape[0],:]
twopca_tract = twopca_results[MATRB.shape[0]:MATRB.shape[0]+MATRT.shape[0],:]
twopca_lights = twopca_results[MATRB.shape[0]+MATRT.shape[0]:MATRB.shape[0]+MATRT.shape[0]+Mean_Pos.shape[0],:]
twopca_meanpos = twopca_results[MATRB.shape[0]+MATRT.shape[0]+Mean_Pos.shape[0]:,:]
print(threedpca_bundle.shape)
print(threedpca_tract.shape)
print(threedpca_lights.shape)
print(threedpca_meanpos.shape)
print(twopca_bundle.shape)
print(twopca_tract.shape)
print(twopca_lights.shape)
print(twopca_meanpos.shape)
'''
twopca_lights = twopca.fit_transform(LIGHTS)
threedpca_lights = threedpca.fit_transform(LIGHTS)
Global = torch.cat((MATRB,MATRT, Mean_Pos),dim=0)
twopca_Global = twopca.transform(Global)
threedpca_Global = threedpca.transform(Global)
twopca_bundle = twopca_Global[:MATRB.shape[0],:]
twopca_tract = twopca_Global[MATRB.shape[0]:MATRB.shape[0]+MATRT.shape[0],:]
twopca_meanpos = twopca_Global[MATRB.shape[0]+MATRT.shape[0]:,:]
threedpca_bundle = threedpca_Global[:MATRB.shape[0],:]
threedpca_tract = threedpca_Global[MATRB.shape[0]:MATRB.shape[0]+MATRT.shape[0],:]
threedpca_meanpos = threedpca_Global[MATRB.shape[0]+MATRT.shape[0]:,:]


color_lights = []
for i in range(57):
    color_lights.append(i)

plt.scatter(twopca_bundle[:,0], twopca_bundle[:,1], c=LAB_1, cmap=l_colors)
plt.scatter(twopca_lights[:,0], twopca_lights[:,1], c="black", marker="x")
# plt.scatter(twopca_tract[:,0], twopca_tract[:,1])
plt.scatter(twopca_tract[:,0], twopca_tract[:,1], c=LAB_2, cmap=l_colors, marker="*")
for i in range(len(twopca_lights)):
    plt.text(twopca_lights[i, 0], twopca_lights[i, 1], str(i), fontsize=12)
plt.scatter(twopca_meanpos[:,0], twopca_meanpos[:,1], c="red", marker="x")
for j in range(len(twopca_meanpos)):
    plt.text(twopca_meanpos[j, 0], twopca_meanpos[j, 1], str(j), fontsize=12, color="red")
# plt.axis([-0.5,1.5,-0.6,0.6])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
color = ax.scatter(threedpca_tract[:,0], threedpca_tract[:,1], threedpca_tract[:,2], c=LAB_2, cmap=l_colors, marker="*")
ax.scatter(threedpca_lights[:,0], threedpca_lights[:,1], threedpca_lights[:,2], linewidths=5, c=color_lights, cmap = l_colors)
color = ax.scatter(threedpca_bundle[:,0], threedpca_bundle[:,1], threedpca_bundle[:,2], c=LAB_1, cmap=l_colors)
# ax.scatter(threedpca_tract[:,0], threedpca_tract[:,1], threedpca_tract[:,2])
for i in range(len(threedpca_lights)):
    ax.text(threedpca_lights[i, 0], threedpca_lights[i, 1], threedpca_lights[i, 2], str(i), fontsize=12)
ax.scatter(threedpca_meanpos[:,0], threedpca_meanpos[:,1], threedpca_meanpos[:,2], c="red", marker="x")
for j in range(len(threedpca_meanpos)):
    ax.text(threedpca_meanpos[j, 0], threedpca_meanpos[j, 1], threedpca_meanpos[j, 2], str(j), fontsize=12, color="red")
# ax.axes.set_xlim3d(left=-0.5, right=1.5)
# ax.axes.set_ylim3d(bottom=-0.6, top=0.6)
# ax.axes.set_zlim3d(bottom=-0.6, top=0.6)
fig.colorbar(color)
plt.show()
missing_list = []
for i in range(57):
    cpt = 0
    for j in range(len(LAB_2)):
        if LAB_2[j] == i:
            cpt += 1
    if cpt>0:
        missing_list.append([i,cpt])
print(missing_list, len(missing_list))





for i in range(len(threedpca_lights)):
    l_i = (LAB_1==i).nonzero().squeeze()
    l_i = [j.item() for j in l_i]
    ax= plt.axes(projection='3d')
    ax.text(threedpca_lights[i, 0], threedpca_lights[i, 1], threedpca_lights[i, 2], str(i), fontsize=12)
    ax.scatter(threedpca_lights[i,0], threedpca_lights[i,1], threedpca_lights[i,2], linewidths=1, c='black')
    # ax.scatter(threedtsne_results[i*205:(i+1)*205,0], threedtsne_results[i*205:(i+1)*205,1], threedtsne_results[i*205:(i+1)*205,2], c=LAB[i*205:(i+1)*205], cmap=l_colors)
    ax.scatter(threedpca_bundle[l_i,0], threedpca_bundle[l_i,1], threedpca_bundle[l_i,2], c=LAB_1[l_i], cmap=l_colors)
    # ax.scatter(threedpca_meanpos[i,0], threedpca_meanpos[i,1], threedpca_meanpos[i,2], c="red", marker="x")
    # if i in missing_list:
        # ax.scatter(threedpca_tract[l_2,0], threedpca_tract[l_2,1], threedpca_tract[l_2,2], c=LAB_2[l_2], cmap=l_colors, marker="*")
    # ax.axes.set_xlim3d(left=-0.5, right=-0.15)
    # ax.axes.set_ylim3d(bottom=-0.3, top=0.2)
    # ax.axes.set_zlim3d(bottom=-0.30, top=0.20)
    plt.show()






# print(asfjdhskgjdhs)





# nb = 0
# list_include=[]
# for i in range(len(twopca_tract)):
#     if twopca_tract[i,0]<-0.15 and twopca_tract[i,1]<0.3:
#         nb+=1
#         list_include.append(i)
# print(len(list_include))

# print("nb_points",nb)

# nb_exclude = 0
# list_exclude = []
# for i in range(len(twopca_tract)):
#     if twopca_tract[i,0]>0.9 and twopca_tract[i,1]<0.0:
#         nb_exclude+=1
#         list_exclude.append(i)
# print("nb_exclude",nb_exclude)
# # print(list_exclude)
# # print(n_lab[10078])
# # print(MATRT.shape)
# # L_D_128D = []
# # seuil128d = 0.2
# # for i in range(len(LIGHTS)):
# #     print(i)
# #     L_i_128D = []
# #     for j in range(len(MATRT)):
# #         distance =0
# #         for a in range(MATRT.shape[1]):
# #             distance += (LIGHTS[i,a]-MATRT[j,a])**2
# #         distance = m.sqrt(distance)
# #         if distance<seuil128d:
# #             L_i_128D.append(j)
# #     L_D_128D.extend(L_i_128D)

# # L_D_128D = sorted(L_D_128D)
# # L_D_128D = list(np.unique(L_D_128D))
# # print("nb_closest points from lights in 128 D ",len(L_D_128D))

# L_distance_l = []
# seuil = 0.07
# for i in range(len(threedpca_lights)):
#     L_i = []
#     for j in range(len(threedpca_tract)):
#         distance = m.sqrt((threedpca_lights[i,0]-threedpca_tract[j,0])**2+(threedpca_lights[i,1]-threedpca_tract[j,1])**2+(threedpca_lights[i,2]-threedpca_tract[j,2])**2)
#         if distance<seuil:
#             L_i.append(j)
#     L_distance_l.extend(L_i)
# # L_distance = 
# L_distance = sorted(L_distance_l)
# # print(L_distance)
# L_distance = list(np.unique(L_distance))
# # print(L_distance)
# print("nb_closest points from lights ",len(L_distance))


# # twopca_tract = twopca_tract[list_include,:]
# # threedpca_tract = threedpca_tract[list_include,:]
# print(LAB_2.shape)
# LAB_2 = LAB_2[L_distance]
# print(LAB_2.shape)
# n_lab = n_lab[L_distance]
# print(n_lab.shape)

# twopca_tract = twopca_tract[L_distance,:]
# threedpca_tract = threedpca_tract[L_distance,:]



# for t in range(len(threedpca_tract)):
#     l_d = []
#     for l in range(len(threedpca_lights)):
#         distance = m.sqrt((threedpca_tract[t,0]-threedpca_lights[l,0])**2+(threedpca_tract[t,1]-threedpca_lights[l,1])**2+(threedpca_tract[t,2]-threedpca_lights[l,2])**2)
#         l_d.append(distance)
#     min_l_d = l_d.index(min(l_d))
#     LAB_2[t] = min_l_d
# print("LAB_2",LAB_2.shape)
# print("LAB_2",LAB_2)
# print(torch.unique(LAB_2))
# print(len(torch.unique(LAB_2)))

# list_i_closed = []
# for i in range(57):
#     L_i_c = []
#     for j in range(len(LAB_2)):
#         if LAB_2[j]==i:
#             L_i_c.append(j)
#     list_i_closed.append(L_i_c)
# print(list_i_closed)
# print(len(list_i_closed))

# Dist_i_closed = []
# for a in range(len(list_i_closed)):
#     threedpca_tract_i = threedpca_tract[list_i_closed[a],:]
#     twopca_tract_i = twopca_tract[list_i_closed[a],:]
#     distance_a = []
#     for i in range(len(threedpca_tract_i)):
#         dist = m.sqrt((threedpca_tract_i[i,0]-threedpca_lights[a,0])**2+(threedpca_tract_i[i,1]-threedpca_lights[a,1])**2+(threedpca_tract_i[i,2]-threedpca_lights[a,2])**2)
#         distance_a.append(dist)
#     Dist_i_closed.append(distance_a)
# # print(Dist_i_closed)
# print(len(Dist_i_closed))
# Mean_distance_closed = []
# Std_distance_closed = []
# for d in range(len(Dist_i_closed)):
#     if len(Dist_i_closed[d])>1:
#         std_d = (sum((x-(sum(Dist_i_closed[d]) / len(Dist_i_closed[d])))**2 for x in Dist_i_closed[d]) / (len(Dist_i_closed[d])-1))**0.5
#         mean_d = np.mean(Dist_i_closed[d])
#     else:
#         std_d = 0
#         mean_d =0
#     Std_distance_closed.append(std_d)
#     Mean_distance_closed.append(mean_d)
# print(Mean_distance_closed)
# print(len(Mean_distance_closed))
# print(Std_distance_closed)
# print(len(Std_distance_closed))
# list_f = []
# L_n_lab = []
# for i in range(len(Dist_i_closed)):
#     L_i_n_lab = []
#     for j in range(len(Dist_i_closed[i])):
#         # print(Dist_i_closed[i][j])
#         # print(Mean_distance_closed[i][j])
#         # print(Std_distance_closed[i][j])
#         # if Dist_i_closed[i][j]>(Mean_distance_closed[i]+Std_distance_closed[i]):
#             # print("coucou")
#             # np.delete(threedpca_tract, list_i_closed[i][j])
#             # np.delete(twopca_tract, list_i_closed[i][j])
#         if Dist_i_closed[i][j]<(Mean_distance_closed[i]+Std_distance_closed[i]):
#             list_f.append(list_i_closed[i][j])
#             L_i_n_lab.append(n_lab[list_i_closed[i][j]])
#     L_n_lab.append(L_i_n_lab)
# print(list_f)
# twopca_tract = twopca_tract[list_f,:]
# threedpca_tract = threedpca_tract[list_f,:]
# print("twopca_tract",twopca_tract.shape)
# print("threedpca_tract",threedpca_tract.shape)
# LAB_2 = LAB_2[list_f]
# print("LAB_2",LAB_2.shape)
# n_lab = n_lab[list_f]
# print("n_lab",n_lab.shape)
# # print(L_n_lab)
# print(len(L_n_lab))
# # file_name = "L_n_lab.pkl"
# # open_file = open(file_name, "wb")
# # pickle.dump(L_n_lab, open_file)
# # open_file.close()
# color_lights = []
# for i in range(57):
#     color_lights.append(i)

# # print("twopca_tract",twopca_tract.shape)
# # plt.scatter(twopca_bundle[:,0], twopca_bundle[:,1], c=LAB_1, cmap=l_colors)
# plt.scatter(twopca_lights[:,0], twopca_lights[:,1], c="black", marker="x")
# # plt.scatter(twopca_tract[:,0], twopca_tract[:,1], alpha=0.2)
# # plt.scatter(twopca_tract[:,0], twopca_tract[:,1], c=LAB_2, cmap=l_colors, alpha=0.2)
# for i in range(len(twopca_lights)):
#     plt.text(twopca_lights[i, 0], twopca_lights[i, 1], str(i), fontsize=12)
# # plt.axis([-0.5,1.5,-0.6,0.6])
# plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# # ax.scatter(threedpca_tract[:,0], threedpca_tract[:,1], threedpca_tract[:,2], alpha=0.2)
# color = ax.scatter(threedpca_tract[:,0], threedpca_tract[:,1], threedpca_tract[:,2], c=LAB_2, cmap=l_colors, alpha=0.2)
# ax.scatter(threedpca_lights[:,0], threedpca_lights[:,1], threedpca_lights[:,2], linewidths=5, c=color_lights, cmap = l_colors)
# # color = ax.scatter(threedpca_bundle[:,0], threedpca_bundle[:,1], threedpca_bundle[:,2], c=LAB_1, cmap=l_colors)
# for i in range(len(threedpca_lights)):
#     ax.text(threedpca_lights[i, 0], threedpca_lights[i, 1], threedpca_lights[i, 2], str(i), fontsize=12)
# # ax.axes.set_xlim3d(left=-0.5, right=1.5)
# # ax.axes.set_ylim3d(bottom=-0.6, top=0.6)
# # ax.axes.set_zlim3d(bottom=-0.6, top=0.6)
# fig.colorbar(color)
# plt.show()

# for i in range(len(threedpca_lights)):
#     l_i = (LAB_1==i).nonzero().squeeze()
#     l_i = [j.item() for j in l_i]
#     # print(l_i)
#     # print(LAB)
#     # print(LAB[l_i])
#     # print(LAB[l_i].shape)
#     # print(threedtsne_results[l_i,:].shape)
#     ax= plt.axes(projection='3d')
#     ax.text(threedpca_lights[i, 0], threedpca_lights[i, 1], threedpca_lights[i, 2], str(i), fontsize=12)
#     ax.scatter(threedpca_lights[i,0], threedpca_lights[i,1], threedpca_lights[i,2], linewidths=1, c='black')
#     # ax.scatter(threedtsne_results[i*205:(i+1)*205,0], threedtsne_results[i*205:(i+1)*205,1], threedtsne_results[i*205:(i+1)*205,2], c=LAB[i*205:(i+1)*205], cmap=l_colors)
#     ax.scatter(threedpca_bundle[l_i,0], threedpca_bundle[l_i,1], threedpca_bundle[l_i,2], c=LAB_1[l_i], cmap=l_colors)

#     ax.axes.set_xlim3d(left=-0.5, right=-0.15)
#     ax.axes.set_ylim3d(bottom=-0.3, top=0.2)
#     ax.axes.set_zlim3d(bottom=-0.30, top=0.20)
#     plt.show()
#     # print(LAB[i*205:(i+1)*205])