import numpy as np
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as colors
from library import utils_lib as utils
import vtk

liste = os.listdir("/CMF/data/timtey/RESULTS/results_contrastive_learning_071823_best")

l_colors = colors.ListedColormap ( np.random.rand (57,3))

matrix2 = [] #should be shape = (56*100,128)
matrix_1 = []
matrix_2 = []
for i in range(len(liste)):
    matrix = torch.load(f"/CMF/data/timtey/RESULTS/results_contrastive_learning_071823_best/{liste[i]}")
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
# LIGHTS = lights.cpu()
MATRB = MATR_1[:,:128]
MATRB = MATRB.cpu()
MATRT = MATR_2[:,:128]
MATRT = MATRT.cpu()

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

for i in range(Mean_Pos.shape[0]):
    Mean_Pos[i] = Mean_Pos[i]/torch.norm(Mean_Pos[i])

L_accept = []
for i in range(MATRT.shape[0]):
    distance = torch.tensor([])
    for j in range(Mean_Pos.shape[0]):
        dist = torch.sqrt(torch.sum((Mean_Pos[j,:]-MATRT[i,:])**2))
        distance = torch.cat((distance, dist.unsqueeze(0)), dim=0)
    min_dist = torch.argmin(distance)
    LAB_2[i] = min_dist
    if distance[min_dist]<0.7:
        L_accept.append(i)

LAB_1 = LAB_1.cpu()
LAB_2 = LAB_2.cpu()
n_lab = n_lab.cpu()
pca = PCA(n_components=3)
GLOBAL = torch.cat((MATRB, MATRT, Mean_Pos), dim=0)
pca_result = pca.fit_transform(GLOBAL)
labeled_results = pca_result[:MATRB.shape[0],:]
tract_results = pca_result[MATRB.shape[0]:MATRB.shape[0]+MATRT.shape[0],:]
mean_pos_results = pca_result[MATRB.shape[0]+MATRT.shape[0]:,:]
tract_results = tract_results[L_accept,:]
LAB_2 = LAB_2[L_accept]
n_lab = n_lab[L_accept,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color = ax.scatter(labeled_results[:,0], labeled_results[:,1], labeled_results[:,2], c=LAB_1, cmap=l_colors)
ax.scatter(mean_pos_results[:,0], mean_pos_results[:,1], mean_pos_results[:,2],c="red", marker="x")
fig.colorbar(color)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color = ax.scatter(labeled_results[:,0], labeled_results[:,1], labeled_results[:,2], c=LAB_1, cmap=l_colors)
ax.scatter(tract_results[:,0], tract_results[:,1], tract_results[:,2],c=LAB_2, cmap=l_colors, marker="x")
ax.scatter(mean_pos_results[:,0], mean_pos_results[:,1], mean_pos_results[:,2],c="red", marker="x")
fig.colorbar(color)
plt.show()

n_lab_subject_id = n_lab[:,0]
uniqu_id = torch.unique(n_lab_subject_id)
bundle = utils.ReadSurf("/CMF/data/timtey/tractography/all/tractogram_deterministic_139233_dg_flip_1.vtp")
L_bundle = []
for i in range(57):
    L_bundle_i = []
    for j in range(LAB_2.shape[0]):
        if LAB_2[j]==i:
            L_bundle_i.append(utils.ExtractFiber_test_lines(bundle, int(n_lab[j,2].item())))
    L_bundle.append(L_bundle_i)
    merge = utils.Merge(L_bundle_i)
    if merge.GetNumberOfCells() > 0 :
        vtk_writer = vtk.vtkXMLPolyDataWriter()
        vtk_writer.SetFileName("/CMF/data/timtey/tractography/all/Test_tract_Slicer/071823_norm/norm_bundle_"+str(i)+".vtp")
        vtk_writer.SetInputData(merge)
        vtk_writer.Write()