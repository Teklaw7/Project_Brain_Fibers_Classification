import vtk
from tools import utils
import pickle
import torch
import numpy as np
import pandas as pd
from matplotlib import colors
import os
path = "/CMF/data/timtey/tractography/all/tractogram_deterministic_139233_dg_flip_1_DTI.vtp"
bundle = utils.ReadSurf(path)
number_cells = bundle.GetNumberOfCells()
print(number_cells)




lights = pd.read_pickle(r'lights_57_3d_on_sphere.pickle')
# liste = os.listdir("/CMF/data/timtey/results_contrastive_loss_combine_loss_tract_cluster_bundle")
# liste = os.listdir("/CMF/data/timtey/results_contrastive_learning_063023_best_model")
liste = os.listdir("/CMF/data/timtey/results_contrastive_learning_071023")
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
    matrix = torch.load(f"/CMF/data/timtey/results_contrastive_learning_071023/{liste[i]}")
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







LABt = torch.load("LABt_071023_mean_pos.pt")


openfile = open("to_accept_01_071023.pkl", "rb")
l_accept = pickle.load(openfile)
openfile.close()
print(LABt.shape)
# output, indices = torch.unique(LABt, return_inverse=True)
# print(output)
# sum = 0
# for i in range(len(output)):
#     print(indices[i].item())    # print(torch.sum(indices==i))
#     sum += indices[i].item()
# print(sum)

# CPT = []
# for i in range(57):
#     cpt_i = 0
#     for j in range(len(LABt)):
#         print(LABt[j].item())
#         if int(LABt[j].item()) == i:
#             cpt_i +=1
#     CPT.append(cpt_i)

# print(CPT)
print(len(l_accept))
print(MATRT.shape)
MATRT = MATRT[l_accept]
LABt = LABt[l_accept]
n_labT = n_labT[l_accept]
Data_labT = Data_labT[l_accept]
print(MATRT.shape)

# for j in range(MATRT.shape[0]):
    # print(n_labT[j][0].item())
L=[]
for i in range(57):
    L.append([])


# for i in range()
for a in range(MATRT.shape[0]):
    lab = int(LABt[a].item())
    L[lab].append(n_labT[a])

# print(L)
# print(len(L))

for l in range(len(L)):
    l_fiber=[]
    for i in range(len(L[l])):
        fiber = utils.ExtractFiber_test(bundle,int(L[l][i][-1].item()))
        l_fiber.append(fiber)
    merge =utils.Merge(l_fiber)
    # vtk_writer = vtk.vtkXMLPolyDataWriter()
    # vtk_writer.SetFileName(f"/CMF/data/timtey/tractography/all/Test_tract_Slicer/test_tracts_slicer_3d/cluster_071023_mean_pos/cluster_{l}.vtp")
    # vtk_writer.SetInputData(merge)
    # vtk_writer.Write()