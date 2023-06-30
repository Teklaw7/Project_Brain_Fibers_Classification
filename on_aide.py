import pandas
import numpy as np
from tools import utils
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.structures import Meshes, join_meshes_as_scene
import vtk
from pytorch3d.renderer.blending import sigmoid_alpha_blend, hard_rgb_blend
import torch
from pytorch3d.renderer import TexturesVertex
from vtk.util.numpy_support import vtk_to_numpy

path = "/home/timtey/Documents/datasets/dataset4/tractography_3.csv"
path2 = "/home/timtey/Documents/datasets/dataset4/tractography_3_test.csv"
df = pandas.read_csv(path)
df2 = pandas.read_csv(path2)
df2["cell_id"] = 0
print(df2)
print(df.iloc[0])
# df2 = df2.append(df.iloc[0])
# df2 = df2.insert(12, 'cell_id', 0)
# print("df2 ",df2)
# print(len(df))
# print(zdjfg)
for i in range(32,len(df)):
    path_bundle = f"{df['surf'][i]}"
    bundle = utils.ReadSurf(path_bundle)
    # print(df['surf'][i])
    nb_cells = bundle.GetNumberOfCells()
    # nb_cells=10
    # print(nb_cells)
    # print(KSJhfdkjhf)
    # print(df.loc[i])
    # print(ajdgf)
    for j in range(nb_cells):
        dd = df.iloc[i]
        dd['cell_id'] = j
        df2 = df2.append(dd)
        # df2['cell_id'][-1] = j
        # print(df2)
        # df2['cell_id'] = j
    
print(df2)
print(len(df2))

df2.to_csv("/home/timtey/Documents/datasets/dataset4/tractography_3_test.csv")
# df2.to_csv("/home/timtey/Documents/Projet_contrastive_double_batch/dataset4/tracts_filtered_train_test_label_to_number_nb_cells_without_missing_all.csv")
